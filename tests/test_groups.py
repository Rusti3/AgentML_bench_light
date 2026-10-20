import pytest

from agentbench.common import read_json,STORE,write_json
from agentbench.groups import manifest,task_ids,select_groups,manifest_sha256,summarize_groups


def test_groups_are_disjoint_balanced_and_independent():
    groups=manifest()['groups']
    assert len(groups)==4 and len(task_ids())==len(set(task_ids()))==16
    assert len({t['family_id'] for g in groups for t in g['tasks']})==16
    for group in groups:
        types=[t['problem_type'] for t in group['tasks']]
        assert types.count('regression')==2
        assert types.count('binary_classification')==1
        assert types.count('multiclass')==1


@pytest.mark.parametrize('value,expected',[(None,16),('all',16),('1',4),('1,3',8),(['1','3'],8),('3,1,3',8)])
def test_group_selection(value,expected):
    assert len(task_ids(value))==expected


@pytest.mark.parametrize('value',['0','5','1,5','all,1','',[]])
def test_invalid_group_selection(value):
    with pytest.raises(ValueError):select_groups(value)


def test_cli_calibration_receives_exact_selected_groups(monkeypatch):
    import agentbench.cli as cli
    import agentbench.calibrate as module
    called=[]
    monkeypatch.setattr(module,'calibrate',lambda *args:called.append(args))
    monkeypatch.setattr('sys.argv',['benchmark.py','calibrate','--groups','1,3'])
    cli.main()
    assert called[0][2]==task_ids('1,3')


@pytest.mark.parametrize('argv',[
    ['calibrate','--groups','1','--tasks','openml_363612'],
    ['run','--agent','example.py','--groups','1','--subset','smoke']])
def test_conflicting_selection_rejected_before_execution(monkeypatch,argv):
    from agentbench.cli import main
    monkeypatch.setattr('sys.argv',['benchmark.py',*argv])
    with pytest.raises(SystemExit) as exc:main()
    assert exc.value.code==2


def test_group_scores_and_partial_selection():
    rows=[]
    for g in manifest()['groups']:
        for t in g['tasks']:
            rows.append({'task_id':t['task_id'],'family_id':t['family_id'],
                         'problem_type':t['problem_type'],'score':10*int(g['id'])})
    from agentbench.metrics import aggregate
    scores=summarize_groups(rows)
    assert aggregate(rows)['overall_score']==25
    assert [scores[k]['overall_score'] for k in ['1','2','3','4']]==[10,20,30,40]
    assert all(g['complete_group_selection'] and g['complete'] for g in scores.values())
    rows[0]['score']=None
    assert summarize_groups(rows)['1']['coverage']==.75
    assert not summarize_groups(rows)['1']['complete']
    assert not summarize_groups(rows[1:])['1']['complete_group_selection']


def test_score_run_keeps_unstarted_cases_in_group_coverage(tmp_path,monkeypatch):
    import agentbench.runner as runner
    monkeypatch.setattr(runner,'STORE',tmp_path)
    tasks=manifest()['groups'][0]['tasks']
    cases=[{**t,'case_id':t['task_id']+'__r0f0','group_id':'1'} for t in tasks]
    path=tmp_path/'runs'/'fixture'
    write_json(path/'manifest.json',{'profile':'local8','subset':'standard','cases':cases,
                                   'groups_manifest_sha256':manifest_sha256()})
    result=runner.score_run('fixture')
    assert result['cases']==4 and result['coverage']==0
    assert result['group_scores']['1']['cases']==4
    assert result['group_scores']['1']['complete'] is False
    saved=read_json(path/'manifest.json');saved['groups_manifest_sha256']='changed'
    write_json(path/'manifest.json',saved)
    with pytest.raises(ValueError,match='composition changed'):runner.score_run('fixture')
