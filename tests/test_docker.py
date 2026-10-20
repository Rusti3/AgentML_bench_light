import http.server
import json
from pathlib import Path
import threading

import pytest

from agentbench.common import ROOT,STORE,read_json,sha256,digest
from agentbench.docker_runtime import check_runtime,launch
from agentbench.gateway import model_gateway
from agentbench.tasks import materialize,select_tasks,selected_splits,case_key
from agentbench.calibrate import calibrate,calibration_path
from agentbench.runner import run_agent
from agentbench.groups import manifest_sha256

pytestmark=pytest.mark.docker


@pytest.fixture(scope='module')
def ready_runtime():
    try:return check_runtime('local8')
    except RuntimeError as exc:pytest.skip(str(exc))


def test_contest_profile_rejected_on_small_engine(ready_runtime):
    if ready_runtime['engine_memory_bytes']>=32*1024**3:
        assert check_runtime('contest32')['limits']['memory_gib']==32
        return
    with pytest.raises(RuntimeError,match='requires 32 GiB'):
        check_runtime('contest32')


def test_network_disabled_and_answers_not_mounted(ready_runtime,tmp_path):
    workspace=tmp_path/'public';workspace.mkdir()
    private=tmp_path/'private';private.mkdir();(private/'answers.csv').write_text('do-not-expose')
    code='''import json,os,socket,pathlib
checks={'private_absent':not os.path.exists('/private/answers.csv'),
        'host_dataset_absent':not os.path.exists('/data/v1/raw'),
        'docker_socket_absent':not os.path.exists('/var/run/docker.sock'),
        'bundled_labels_absent':not os.path.exists('/opt/benchsite/lib/python3.12/site-packages/sklearn/datasets/data/breast_cancer.csv'),
        'plotly_dataset_absent':not os.path.exists('/opt/benchsite/lib/python3.12/site-packages/plotly/package_data/datasets/iris.csv.gz'),
        'test_openml_tables_absent':not list(pathlib.Path('/opt/benchsite/lib/python3.12/site-packages/sklearn/datasets/tests/data').rglob('*.arff.gz')),
        'memory_limit_8g':pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip()==str(8*1024**3),
        'cpu_limit_8':pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip()=='800000 100000',
        'swap_disabled':pathlib.Path('/sys/fs/cgroup/memory.swap.max').read_text().strip()=='0'}
try:
 socket.create_connection(('1.1.1.1',80),timeout=2)
 checks['internet_blocked']=False
except OSError: checks['internet_blocked']=True
open('probe.json','w').write(json.dumps(checks))
assert all(checks.values()),checks
'''
    result=launch(workspace,'local8',['python','-c',code],30,tmp_path/'logs')
    assert result['exit_code']==0, (tmp_path/'logs'/'stderr.log').read_text()
    assert all(json.loads((workspace/'probe.json').read_text()).values())


def test_unix_gateway_allows_model_route_only_and_hides_key(ready_runtime,tmp_path):
    seen=[]
    class Mock(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            seen.append((self.path,self.headers.get('Authorization'),self.headers.get('x-openai-actor-authorization')))
            body=b'{"data":[{"id":"mock-model"}]}'
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path,self.headers.get('Authorization'),self.headers.get('x-openai-actor-authorization'),body))
            payload=b'data: {"type":"response.completed"}\n\ndata: [DONE]\n\n'
            self.send_response(200);self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    server=http.server.ThreadingHTTPServer(('0.0.0.0',0),Mock)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    workspace=tmp_path/'public';workspace.mkdir()
    code='''import httpx,os,json
assert 'integration-upstream-secret' not in str(dict(os.environ))
base=os.environ['OPENAI_BASE_URL']
with httpx.Client(timeout=15) as c:
 r=c.get(base+'/models');assert r.status_code==200,(r.status_code,r.text)
 assert r.json()['data'][0]['id']=='mock-model'
 assert c.get(base+'/files').status_code==403
 assert c.get(base+'/responses').status_code==405
 assert c.post(base+'/models',json={}).status_code==405
 with c.stream('POST',base+'/responses',json={'model':'mock-model','input':'ping','stream':True}) as stream:
  assert stream.status_code==200
  assert 'response.completed' in ''.join(stream.iter_text())
open('gateway_probe.json','w').write(json.dumps({'allowed_model_route':True,'other_routes_blocked':True,'upstream_secret_hidden':True}))
'''
    try:
        with model_gateway(f'http://host.docker.internal:{server.server_port}/v1',api_key='integration-upstream-secret',
                           headers='{"x-openai-actor-authorization":"integration-actor"}') as gateway:
            result=launch(workspace,'local8',['python','/entry.py','python','-c',code],45,tmp_path/'logs',
                          extra_mounts=[(ROOT/'agentbench'/'entry_worker.py','/entry.py')],
                          ipc_mounts=gateway['mounts'],environment=gateway['environment'])
        assert result['exit_code']==0,(tmp_path/'logs'/'stderr.log').read_text()
        assert seen==[('/v1/models','Bearer integration-upstream-secret','integration-actor'),
                      ('/v1/responses','Bearer integration-upstream-secret','integration-actor',
                       {'model':'mock-model','input':'ping','stream':True})]
    finally:server.shutdown();server.server_close()


def test_valid_submission_survives_timeout(ready_runtime,tmp_path):
    workspace=tmp_path/'public';workspace.mkdir()
    code="import pathlib,time; pathlib.Path('submission.csv').write_text('id,target\\na,1\\n'); time.sleep(60)"
    result=launch(workspace,'local8',['python','-c',code],8,tmp_path/'logs')
    assert result['timed_out'] and (workspace/'submission.csv').exists()
    truth=tmp_path/'truth.csv';truth.write_text('id,target\na,1\n')
    from agentbench.metrics import validate_and_score
    assert validate_and_score(workspace/'submission.csv',truth,{'problem_type':'regression'})['metrics']['rmse']==0


def test_real_reference_and_agent_classification_and_regression(ready_runtime):
    available=select_tasks('standard')
    chosen=[]
    for kind in ['regression','classification']:
        pool=[t for t in available if (t['problem_type']=='regression')==(kind=='regression')
              and t.get('rows',0)<5000 and t.get('feature_count',0)<=50]
        assert pool,'Need downloaded/audited small cases for integration'
        chosen.append(min(pool,key=lambda t:(t['rows'],t['id'])))
    ids=','.join(t['id'] for t in chosen)
    manifest=calibrate(ids=ids,profile='local8')
    assert all(c['status'] in ['calibrated','uncalibratable'] for c in manifest['cases'])
    summary=run_agent(ROOT/'examples'/'solution.py',ids=ids,profile='local8',
                      run_id='integration_dummy_v1_'+digest([ids,sha256(ROOT/'runtime'/'linux_manifest.json'),manifest_sha256()])[:10],
                      no_llm=True,timeout=120)
    assert all(r['status']=='completed' for r in summary['results'])
    for task in chosen:
        split=next(selected_splits(task));key=case_key(task,split)
        reference=read_json(calibration_path('local8',key))
        second=STORE/'private'/'reproducibility'/key
        truth=STORE/'private'/'reproducibility_truth'/key
        materialize(task,split,second,truth)
        result=launch(second,'local8',['python','/reference.py'],120,
                      STORE/'private'/'reproducibility_logs'/key,
                      extra_mounts=[(ROOT/'agentbench'/'reference_worker.py','/reference.py')])
        assert result['exit_code']==0
        assert sha256(second/'submission.csv')==reference['prediction_sha256']
