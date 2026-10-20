import hashlib
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from agentbench.audit import check_split,license_check
from agentbench.download import Downloader
from agentbench.metrics import aggregate,normalized_score,validate_and_score
from agentbench.tasks import materialize,public_id


@pytest.mark.parametrize('kind,value,baseline,expected',[
    ('regression',20,40,50),('regression',50,40,0),('regression',0,40,100),
    ('classification',.8,.5,60),('classification',.4,.5,0),('classification',1,.5,100),
    ('classification',1,1,None),('regression',1,0,None)])
def test_pdf_score(kind,value,baseline,expected):
    got=normalized_score(value,baseline,kind)
    if expected is None:assert got is None
    else:assert got==pytest.approx(expected)


def test_family_aggregation_prevents_duplicate_weight():
    rows=[{'task_id':'a','family_id':'x','problem_type':'binary_classification','score':100},
          {'task_id':'a','family_id':'x','problem_type':'binary_classification','score':100},
          {'task_id':'a_variant','family_id':'x','problem_type':'binary_classification','score':100},
          {'task_id':'b','family_id':'y','problem_type':'multiclass','score':0},
          {'task_id':'c','family_id':'z','problem_type':'regression','score':40}]
    result=aggregate(rows)
    assert result['category_scores']['classification']==50
    assert result['overall_score']==45


def test_partial_coverage_not_complete():
    rows=[{'task_id':'a','family_id':'x','problem_type':'regression','score':None},
          {'task_id':'b','family_id':'y','problem_type':'regression','score':0}]
    result=aggregate(rows)
    assert result['coverage']==.5 and not result['complete']
    assert result['overall_score'] is None


def test_group_and_temporal_split_checks():
    tr=np.array([0,1]);te=np.array([2,3])
    assert check_split(tr,te,4,groups=np.array([0,1,1,2]))==['train_test_group_overlap']
    assert check_split(tr,te,4,times=np.array([1,3,2,4]),regime='temporal')==['temporal_order_violation']
    assert 'train_test_row_overlap' in check_split(tr,np.array([1,3]),4)
    assert 'test_index_out_of_bounds' in check_split(tr,np.array([9]),4)


def test_public_is_not_a_license():
    assert license_check('Public')=='ambiguous'
    assert license_check('CC BY 4.0')=='recognized'
    assert license_check('CC BY-NC 4.0')=='restricted'
    assert license_check('CC BY NC 3.0')=='restricted'


@pytest.fixture
def scoring(tmp_path):
    truth=tmp_path/'truth.csv'
    pd.DataFrame({'id':['a','b'],'target':['A','B']}).to_csv(truth,index=False)
    meta={'problem_type':'multiclass','classes':['A','B','C']}
    return tmp_path,truth,meta


@pytest.mark.parametrize('pred',[
    {'id':['b','a'],'target':['A','B']},
    {'id':['a','a'],'target':['A','B']},
    {'id':['a'],'target':['A']},
    {'id':['a','b'],'target':['A','OUTSIDE']},
    {'id':['a','b'],'target':['A','']},
])
def test_invalid_submissions(scoring,pred):
    root,truth,meta=scoring;path=root/'submission.csv';pd.DataFrame(pred).to_csv(path,index=False)
    with pytest.raises(ValueError):validate_and_score(path,truth,meta)


def test_perfect_active_classes_f1_is_one(scoring):
    root,truth,meta=scoring
    result=validate_and_score(truth,truth,meta)
    assert result['metrics']['macro_f1']==1


def test_regression_rejects_infinity(scoring):
    root,truth,_=scoring
    pd.DataFrame({'id':['a','b'],'target':[1,2]}).to_csv(truth,index=False)
    path=root/'submission.csv';pd.DataFrame({'id':['a','b'],'target':[1,np.inf]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Nonfinite'):validate_and_score(path,truth,{'problem_type':'regression'})


def test_download_resumes_and_verifies_hash(tmp_path):
    content=b'0123456789'*1000;requested=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            offset=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0]);requested.append(offset)
            body=content[offset:]
            self.send_response(206 if offset else 200)
            if offset:self.send_header('Content-Range',f'bytes {offset}-{len(content)-1}/{len(content)}')
            self.send_header('Content-Length',str(len(body)));self.send_header('ETag','"pinned"');self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/data';path=tmp_path/'data.bin'
    path.with_name('data.bin.part').write_bytes(content[:100])
    path.with_name('data.bin.part.json').write_text(json.dumps({'url':url,'etag':'"pinned"'}))
    dl=Downloader(attempts=1)
    try:
        receipt=dl.get(url,path,expected_sha=hashlib.sha256(content).hexdigest())
        assert requested==[100] and path.read_bytes()==content
        dl.get(url,path,expected_sha=receipt['sha256']);assert requested==[100]
    finally:dl.close();server.shutdown();server.server_close()


def test_materialization_hides_labels_and_preserves_literal_na(tmp_path,monkeypatch):
    import agentbench.tasks as module
    frame=pd.DataFrame({'id':['original1','original2','original3','original4'],
                        'text':['NA','',None,'hello'],'outcome':['A','B','A','B']})
    table=tmp_path/'data.parquet';frame.to_parquet(table,index=False)
    from agentbench.common import sha256
    task={'id':'fixture','name':'Fixture','adapter':'openml','target':'outcome',
          'problem_type':'binary_classification','feature_columns':['id','text'],
          'table_path':str(table),'table_sha256':sha256(table),'classes':['A','B'],
          'split_regime':'iid','feature_metadata':[]}
    monkeypatch.setattr(module,'resolve',lambda p:Path(p))
    public=tmp_path/'public';private=tmp_path/'private'
    meta=materialize(task,{'repeat':0,'fold':0,'train':np.array([0,1]),'test':np.array([2,3])},public,private)
    test=pd.read_csv(public/'test.csv',keep_default_na=False,na_values=[meta['missing_value_token']])
    train=pd.read_csv(public/'train.csv',keep_default_na=False,na_values=[meta['missing_value_token']])
    assert 'target' not in test and 'feature__id' in test
    assert train.text.tolist()==['NA','']
    assert test.text.isna().tolist()==[True,False]
    assert not (public/'answers.csv').exists() and (private/'answers.csv').exists()
    assert test.id.tolist()==[public_id(task['table_sha256'],p) for p in [2,3]]
    table.write_bytes(b'changed canonical source')
    with pytest.raises(ValueError,match='Canonical table checksum'):
        materialize(task,{'repeat':0,'fold':0,'train':np.array([0,1]),'test':np.array([2,3])},public,private)


@pytest.fixture
def audited_fixture(tmp_path,monkeypatch):
    import agentbench.audit as module
    from agentbench.common import sha256
    frame=pd.DataFrame({'AssignedRoomType':['A','B','B','A'],'x':[3,4,5,6],'y':[0,1,0,1]})
    table=tmp_path/'table.parquet';frame.to_parquet(table,index=False)
    split=tmp_path/'split.npz';np.savez_compressed(split,train=[0,1],test=[2,3])
    index=tmp_path/'index.json';index.write_text(json.dumps([{'repeat':0,'fold':0,'path':str(split),'sha256':sha256(split)}]))
    task={'id':'fixture','name':'fixture','target':'y','table_path':str(table),'table_sha256':sha256(table),
          'split_index':str(index),'split_regime':'iid','problem_type':'classification','license':'CC BY 4.0'}
    monkeypatch.setattr(module,'resolve',lambda p:Path(p))
    return module,task,frame,table,split


def test_known_leak_is_quarantined(audited_fixture):
    module,task,*_=audited_fixture;task['name']='hotel_booking_demand'
    result=module.audit_one(task)
    assert result['status']=='quarantined' and result['known_leak']['columns']


def test_exact_target_feature_is_quarantined(audited_fixture):
    from agentbench.common import sha256
    module,task,frame,table,_=audited_fixture
    frame['leaked']=frame['y'];frame.to_parquet(table,index=False);task['table_sha256']=sha256(table)
    result=module.audit_one(task)
    assert result['status']=='quarantined' and result['target_copies']==['leaked']


def test_missing_target_and_tampered_split_rejected(audited_fixture):
    module,task,_,_,split=audited_fixture
    task['target']='not-a-column'
    with pytest.raises(ValueError,match='target absent'):module.audit_one(task)
    task['target']='y';split.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Split checksum'):module.audit_one(task)


def test_repository_name_is_not_data_license():
    assert license_check('UCI Repository')=='ambiguous'
    assert license_check('permission unspecified')=='ambiguous'


def test_run_rejects_excessive_time_and_score_path_escape():
    from agentbench.runner import run_agent,score_run
    with pytest.raises(ValueError,match='10800'):run_agent('unused.py',timeout=10801)
    with pytest.raises(ValueError,match='Invalid run ID'):score_run('../private')


def test_compute_lock_resumes_stale_job_and_rejects_parallel(tmp_path,monkeypatch):
    import os
    import agentbench.docker_runtime as module
    from agentbench.common import write_json,read_json
    monkeypatch.setattr(module,'STORE',tmp_path)
    path=tmp_path/'compute.lock.json'
    write_json(path,{'pid':os.getpid(),'create_time':1})
    with module.compute_lock():
        assert read_json(path)['pid']==os.getpid()
        with pytest.raises(RuntimeError,match='active'):
            with module.compute_lock():pass
    assert not path.exists()


def test_equal_values_with_different_targets_are_separate_cases(tmp_path,monkeypatch):
    import agentbench.audit as module
    from agentbench.common import read_json
    tasks=[{'id':f'case_{i}','target':target,'problem_type':'regression','status':'verified',
            'audit':{'status':'verified','reasons':[],'fingerprint':'same-data',
                     'protocol_fingerprint':'same-protocol','task_updates':{}}}
           for i,target in enumerate(['target_a','target_b'])]
    monkeypatch.setattr(module,'STORE',tmp_path)
    monkeypatch.setattr(module,'effective_tasks',lambda:tasks)
    monkeypatch.setattr(module,'audit_path',lambda tid:tmp_path/(tid+'.json'))
    module.resolve_families_and_duplicates()
    assert all(read_json(tmp_path/(t['id']+'.json'))['status']=='verified' for t in tasks)
