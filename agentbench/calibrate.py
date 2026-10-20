from __future__ import annotations

from pathlib import Path
import shutil

from .common import ROOT,STORE,digest,event,read_json,sha256,write_json
from .docker_runtime import check_runtime,compute_lock,launch
from .metrics import normalized_score,validate_and_score
from .tasks import case_key,case_signature,materialize,select_tasks,selected_splits


def calibration_path(profile,key):
    return STORE/'calibration'/profile/key/'result.json'


def reference_signature(task,split,profile):
    return digest({'case':case_signature(task,split),'baseline_worker':sha256(Path(__file__).with_name('reference_worker.py')),
                   'runtime':sha256(ROOT/'runtime'/'linux_manifest.json'),'profile':profile,'reference':'catboost_v1'})


def calibrate_one(task,split,profile,retry=False):
    key=case_key(task,split)
    result_path=calibration_path(profile,key);folder=result_path.parent
    signature=reference_signature(task,split,profile)
    old=read_json(result_path,{})
    if old.get('signature')==signature and old.get('status')=='calibrated':
        if sha256(folder/'submission.csv')!=old.get('prediction_sha256'):
            raise RuntimeError('Frozen calibration prediction changed')
        if old.get('family_id')!=task['family_id']:
            old['family_id']=task['family_id'];write_json(result_path,old)
        return old
    if old.get('signature') and old['signature']!=signature and old.get('status')=='calibrated':
        raise RuntimeError('A published calibration is immutable; create a new benchmark version')
    if old.get('status') in ['failed','uncalibratable'] and not retry:return old
    workspace=STORE/'private'/'calibration_work'/profile/key
    truth=STORE/'private'/'calibration_truth'/profile/key
    meta=materialize(task,split,workspace,truth)
    (workspace/'submission.csv').unlink(missing_ok=True)
    result={'task_id':task['id'],'case_id':key,'profile':profile,'signature':signature,'status':'running',
            'family_id':task['family_id'],'problem_type':task['problem_type'],'repeat':split['repeat'],'fold':split['fold']}
    write_json(result_path,result)
    run=launch(workspace,profile,['python','/worker/reference_worker.py'],600,folder,
               extra_mounts=[(Path(__file__).with_name('reference_worker.py'),'/worker/reference_worker.py')],
               name='agentbench-ref-'+digest([profile,key])[:20])
    result['execution']=run
    try:
        if run['exit_code']!=0 or run['timed_out']:raise ValueError('Reference failed or exceeded 600 seconds')
        scored=validate_and_score(workspace/'submission.csv',truth/'answers.csv',meta)
        result.update(scored)
        metric='rmse' if task['problem_type']=='regression' else 'macro_f1'
        result['baseline_metric']=scored['metrics'][metric]
        result['status']='calibrated' if normalized_score(scored['metrics'][metric],scored['metrics'][metric],
                                                        'regression' if metric=='rmse' else 'classification') is not None else 'uncalibratable'
        if result['status']=='uncalibratable':result['reason']='Baseline metric makes PDF normalization undefined'
        shutil.copy2(workspace/'submission.csv',folder/'submission.csv')
        shutil.copy2(workspace/'reference_model.json',folder/'reference_model.json')
        result['prediction_sha256']=sha256(folder/'submission.csv')
    except Exception as exc:
        result.update(status='failed',reason=str(exc))
    write_json(result_path,result)
    # Generated inputs are reconstructible; avoid multiplying data across calibrations.
    for name in ['train.csv','test.csv','sample_submission.csv']:
        (workspace/name).unlink(missing_ok=True)
    event('calibration',task=task['id'],case=key,status=result['status'],metric=result.get('baseline_metric'),
          seconds=run['seconds'])
    return result


def calibrate(subset='standard',profile='local8',ids=None,retry=False,limit=None):
    runtime=check_runtime(profile)
    tasks=select_tasks(subset,ids)
    if limit:tasks=tasks[:limit]
    manifest_name=subset if not ids and not limit else subset+'__selection_'+digest([t['id'] for t in tasks])[:12]
    manifest={'profile':profile,'subset':subset,'runtime':runtime,'cases':[]}
    with compute_lock():
        for task in tasks:
            for split in selected_splits(task,subset):
                result=calibrate_one(task,split,profile,retry)
                manifest['cases'].append({'case_id':result['case_id'],'task_id':task['id'],'status':result['status'],
                                           'signature':result['signature']})
                write_json(STORE/'calibration'/profile/f'{manifest_name}_manifest.json',manifest)
    return manifest
