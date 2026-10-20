from __future__ import annotations

from datetime import datetime,timezone
from pathlib import Path
import re
import shutil
import os

from .calibrate import calibration_path,reference_signature
from .common import ROOT,STORE,digest,event,read_json,sha256,write_json
from .docker_runtime import check_runtime,compute_lock,launch
from .gateway import model_gateway
from .metrics import aggregate,normalized_score,validate_and_score
from .groups import task_group_map,manifest_sha256,summarize_groups
from .tasks import case_key,case_signature,materialize,select_tasks,selected_splits


def agent_files(agent):
    agent=Path(agent).resolve()
    if not agent.exists():raise ValueError('Agent path not found')
    if agent.is_symlink():raise ValueError('Symlink agent bundles are not permitted')
    if agent.is_file():
        if agent.suffix!='.py':raise ValueError('Single-file agent must be a Python script')
        return [(agent,'solution.py')]
    if not (agent/'solution.py').is_file():raise ValueError('Agent directory needs solution.py')
    if ROOT.resolve().is_relative_to(agent) or agent.is_relative_to(STORE.resolve()):
        raise ValueError('Use a separate agent bundle, not the benchmark/data root')
    files=[]
    for path in agent.rglob('*'):
        if any(p in ['.git','.venv','__pycache__'] for p in path.relative_to(agent).parts):continue
        if path.is_symlink():raise ValueError('Agent bundle contains a symlink')
        if path.is_file():files.append((path,path.relative_to(agent).as_posix()))
    return sorted(files,key=lambda x:x[1])


def run_agent(agent,subset='standard',profile='local8',ids=None,run_id=None,timeout=10800,
              upstream=None,no_llm=False,model=None,environment=None):
    if not 0<timeout<=10800:raise ValueError('Agent timeout must be between 1 and 10800 seconds')
    runtime=check_runtime(profile)
    files=agent_files(agent)
    agent_hash=digest([(rel,sha256(p)) for p,rel in files])
    tasks=select_tasks(subset,ids)
    run_id=run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+agent_hash[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',run_id):raise ValueError('Invalid run ID')
    folder=STORE/'runs'/run_id;folder.mkdir(parents=True,exist_ok=True)
    cases=[(task,split) for task in tasks for split in selected_splits(task,subset)]
    configured_upstream=(upstream or os.environ.get('BENCH_UPSTREAM_BASE_URL') or
                         os.environ.get('OPENAI_BASE_URL')) if not no_llm else None
    identity=digest({'agent':agent_hash,'profile':profile,'cases':[case_signature(t,s) for t,s in cases],
                     'timeout':timeout,'model':model,'upstream':configured_upstream,
                     'runtime':runtime['image_id'],
                     'runtime_packages':sha256(ROOT/'runtime'/'linux_manifest.json'),
                     'groups_manifest':manifest_sha256()})
    existing=read_json(folder/'manifest.json',{})
    if existing and existing.get('identity')!=identity:raise ValueError('Cannot resume a run with different inputs/configuration')
    manifest={'run_id':run_id,'identity':identity,'agent_sha256':agent_hash,'profile':profile,'subset':subset,
              'runtime':runtime,'timeout_seconds':timeout,'model':model,
              'groups_manifest_sha256':manifest_sha256(),
              'cases':[{'case_id':case_key(t,s),'task_id':t['id'],'family_id':t['family_id'],
                        'problem_type':t['problem_type'],'case_signature':case_signature(t,s),
                        'group_id':task_group_map()[t['id']]} for t,s in cases]}
    write_json(folder/'manifest.json',manifest)
    with compute_lock():
        for task,split in cases:
            key=case_key(task,split);case_folder=folder/'cases'/key
            result_file=case_folder/'result.json'
            old=read_json(result_file,{})
            if old.get('status') in ['completed','invalid_submission']:
                event('run_resume_skip',case=key);continue
            workspace=STORE/'private'/'agent_work'/run_id/key
            truth=STORE/'private'/'agent_truth'/run_id/key
            meta=materialize(task,split,workspace,truth)
            # Mounting only this workspace makes sibling answers and datasets inaccessible.
            bundle=workspace/'agent';bundle.mkdir(exist_ok=True)
            for source,rel in files:
                target=bundle/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            (workspace/'submission.csv').unlink(missing_ok=True)
            result={'case_id':key,'task_id':task['id'],'family_id':task['family_id'],
                    'problem_type':task['problem_type'],'status':'running','profile':profile,
                    'case_signature':case_signature(task,split),'score':None}
            write_json(result_file,result)
            try:
                with model_gateway(base_url=upstream if not no_llm else None) if not no_llm else _no_gateway() as gateway:
                    env={**(environment or {}),**gateway['environment']}
                    if model:env['OPENAI_MODEL']=model
                    execution=launch(workspace,profile,['python','/trusted/entry_worker.py',
                                     'python','/workspace/agent/solution.py','/workspace/task.md'],timeout,case_folder,
                                     extra_mounts=[(Path(__file__).with_name('entry_worker.py'),'/trusted/entry_worker.py')],
                                     ipc_mounts=gateway['mounts'],environment=env,
                                     name='agentbench-run-'+digest([run_id,key])[:20])
                result['execution']=execution
                # The PDF evaluates an already saved valid file even if the process times out.
                try:
                    scored=validate_and_score(workspace/'submission.csv',truth/'answers.csv',meta)
                    result.update(scored);result['status']='completed'
                    shutil.copy2(workspace/'submission.csv',case_folder/'submission.csv')
                    result['prediction_sha256']=sha256(case_folder/'submission.csv')
                except ValueError as exc:
                    result.update(status='invalid_submission',valid=False,error=str(exc))
            except Exception as exc:
                result.update(status='infrastructure_error',error=str(exc))
            baseline=read_json(calibration_path(profile,key),{})
            if baseline.get('status')=='calibrated' and baseline.get('signature')==reference_signature(task,split,profile):
                if result['status']=='invalid_submission':result['score']=0.0
                elif result['status']=='completed':
                    primary='rmse' if task['problem_type']=='regression' else 'macro_f1'
                    result['score']=normalized_score(result['metrics'][primary],baseline['baseline_metric'],
                                                     'regression' if primary=='rmse' else 'classification')
                result['baseline_signature']=baseline['signature']
            else:result['normalization_unavailable']='No valid frozen calibration for this split and profile'
            write_json(result_file,result)
            event('run',case=key,status=result['status'],score=result['score'])
            # Input tables can be reconstructed exactly from immutable source data.
            for name in ['train.csv','test.csv','sample_submission.csv']:(workspace/name).unlink(missing_ok=True)
    return score_run(run_id)


def score_run(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',run_id):raise ValueError('Invalid run ID')
    folder=STORE/'runs'/run_id
    manifest=read_json(folder/'manifest.json')
    if not manifest:raise ValueError('Unknown run ID')
    if manifest.get('groups_manifest_sha256')!=manifest_sha256():
        raise ValueError('Group composition changed after the run was created')
    rows=[]
    for case in manifest['cases']:
        result=read_json(folder/'cases'/case['case_id']/'result.json',{})
        if result.get('prediction_sha256'):
            if sha256(folder/'cases'/case['case_id']/'submission.csv')!=result['prediction_sha256']:
                raise ValueError('Saved submission changed after evaluation')
        rows.append({**case,'score':result.get('score'),'status':result.get('status','not_started')})
    summary={'run_id':run_id,'profile':manifest['profile'],'subset':manifest['subset'],**aggregate(rows),
             'group_scores':summarize_groups(rows),'results':rows}
    write_json(folder/'summary.json',summary)
    return summary


from contextlib import contextmanager
@contextmanager
def _no_gateway():
    yield {'mounts':[],'environment':{}}
