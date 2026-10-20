from contextlib import contextmanager
import os
from pathlib import Path
import time
import urllib.parse
import uuid

from .common import ROOT
from .docker_runtime import IMAGE,docker


@contextmanager
def model_gateway(base_url=None,api_key=None,headers=None):
    """The agent stays --network=none; only a RAM-backed Unix socket is shared.

    This avoids giving a supposedly internal Docker network access to host services.
    Secrets live only in the proxy container, outside the agent PID and mount namespaces.
    """
    base_url=base_url or os.environ.get('BENCH_UPSTREAM_BASE_URL') or os.environ.get('OPENAI_BASE_URL')
    if not base_url:
        yield {'mounts':[],'environment':{}};return
    parsed=urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ['http','https'] or parsed.username or parsed.password:
        raise ValueError('Upstream must be a configured HTTP(S) URL without embedded credentials')
    if parsed.hostname in ['localhost','127.0.0.1']:
        host='host.docker.internal'+(':'+str(parsed.port) if parsed.port else '')
        base_url=urllib.parse.urlunsplit((parsed.scheme,host,parsed.path,'',''))
    token=uuid.uuid4().hex[:16];volume='agentbench-ipc-'+token;container='agentbench-gateway-'+token
    made=docker(['volume','create','--label','agentbench.owned=true','--driver','local',
                 '--opt','type=tmpfs','--opt','device=tmpfs','--opt','o=size=16m',volume],timeout=30)
    if made.returncode:raise RuntimeError('Cannot create RAM-backed model gateway IPC')
    env=os.environ.copy()
    env['BENCH_UPSTREAM_BASE_URL']=base_url
    env['BENCH_UPSTREAM_API_KEY']=api_key if api_key is not None else os.environ.get('OPENAI_API_KEY','')
    env['BENCH_UPSTREAM_HEADERS']=headers if headers is not None else os.environ.get('BENCH_UPSTREAM_HEADERS','{}')
    args=['run','-d','--name',container,'--label','agentbench.owned=true','--cpus','0.25','--memory','256m',
          '--cap-drop','ALL','--security-opt','no-new-privileges','--read-only',
          '--mount',f'type=volume,source={volume},target=/ipc',
          '--mount',f'type=bind,source={ROOT/"runtime"/"linux"},target=/opt/benchsite,readonly',
          '--mount',f'type=bind,source={Path(__file__).with_name("gateway_worker.py")},target=/gateway.py,readonly',
          '--env','BENCH_UPSTREAM_BASE_URL','--env','BENCH_UPSTREAM_API_KEY','--env','BENCH_UPSTREAM_HEADERS',
          IMAGE,'python','/gateway.py']
    try:
        started=docker(args,timeout=30,env=env)
        if started.returncode:raise RuntimeError('Model gateway failed to start: '+started.stderr[-500:])
        for _ in range(30):
            log=docker(['logs',container],timeout=10)
            if 'MODEL_GATEWAY_READY' in log.stdout:break
            time.sleep(.2)
        else:raise RuntimeError('Model gateway startup timeout')
        yield {'mounts':[(volume,'/llm')],
               'environment':{'BENCH_LLM_GATEWAY':'1','OPENAI_BASE_URL':'http://127.0.0.1:8090/v1',
                              'OPENAI_API_KEY':'agentbench-local-gateway'},
               'container':container,'volume':volume}
    finally:
        docker(['rm','-f',container],timeout=30)
        docker(['volume','rm',volume],timeout=30)
