from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import psutil

from .common import ROOT,STORE,PROFILES,read_json,write_json

IMAGE='agent-benchmark-base:v1'


def docker(args,**kw):
    return subprocess.run(['docker',*args],capture_output=True,text=True,encoding='utf-8',errors='replace',**kw)


def check_runtime(profile):
    if profile not in PROFILES:raise ValueError('Unknown resource profile')
    info=docker(['info','--format','{{json .}}'],timeout=30)
    if info.returncode:raise RuntimeError('Linux Docker engine unavailable: '+info.stderr[-1000:])
    data=json.loads(info.stdout)
    required=PROFILES[profile]
    if data.get('OSType')!='linux':raise RuntimeError('Linux Docker required')
    if data['MemTotal']<required['memory_gib']*1024**3:
        raise RuntimeError(f'Profile {profile} requires {required["memory_gib"]} GiB; engine has {data["MemTotal"]/1024**3:.2f} GiB. Profile not certified.')
    if data['NCPU']<required['cpus']:raise RuntimeError('Insufficient Docker CPUs')
    image=docker(['image','inspect',IMAGE,'--format','{{.Id}}'],timeout=15)
    if image.returncode:raise RuntimeError('Run setup_runtime.ps1 first')
    manifest=ROOT/'runtime'/'linux_manifest.json'
    if not manifest.exists():raise RuntimeError('Linux package runtime has not passed import checks')
    frozen=read_json(manifest)
    if frozen.get('image_id') and frozen['image_id']!=image.stdout.strip():
        raise RuntimeError('Docker image differs from frozen runtime; restore runtime/base_image.tar')
    return {'image_id':image.stdout.strip(),'engine_cpus':data['NCPU'],'engine_memory_bytes':data['MemTotal'],
            'profile':profile,'limits':required}


@contextmanager
def compute_lock():
    path=STORE/'compute.lock.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        previous=read_json(path,{})
        alive=psutil.pid_exists(previous.get('pid',-1))
        if alive and previous.get('create_time'):
            try:alive=abs(psutil.Process(previous['pid']).create_time()-previous['create_time'])<.01
            except psutil.NoSuchProcess:alive=False
        if alive:
            raise RuntimeError('Another benchmark compute job is active')
        path.unlink()
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'create_time':psutil.Process().create_time()},f)
        yield
    finally:
        if read_json(path,{}).get('pid')==os.getpid():path.unlink(missing_ok=True)


def launch(workspace,profile,command,timeout,log_dir,*,network='none',extra_mounts=None,
           environment=None,name=None,readonly_root=True,ipc_mounts=None):
    """Only the public case and explicitly trusted runtime files are mounted."""
    limits=PROFILES[profile]
    (Path(workspace)/'.home').mkdir(exist_ok=True)
    log_dir=Path(log_dir);log_dir.mkdir(parents=True,exist_ok=True)
    name=name or 'agentbench-'+uuid.uuid4().hex[:16]
    existing=docker(['inspect',name,'--format','{{json .Config.Labels}}'],timeout=15)
    if existing.returncode==0:
        labels=json.loads(existing.stdout or '{}') or {}
        if labels.get('agentbench.owned')!='true':raise RuntimeError('Container name belongs to another application')
        docker(['rm','-f',name],timeout=30)
    args=['run','--name',name,'--label','agentbench.owned=true','--cpus',str(limits['cpus']),
          '--memory',f'{limits["memory_gib"]}g','--memory-swap',f'{limits["memory_gib"]}g',
          '--pids-limit','512','--network',network,'--cap-drop','ALL','--security-opt','no-new-privileges',
          '--mount',f'type=bind,source={Path(workspace).resolve()},target=/workspace',
          '--mount',f'type=bind,source={ROOT/"runtime"/"linux"},target=/opt/benchsite,readonly',
          '--tmpfs','/tmp:rw,noexec,nosuid,size=512m',
          '--env','HOME=/workspace/.home','--env','OMP_NUM_THREADS=8','--env','MKL_NUM_THREADS=8',
          '--env','OPENBLAS_NUM_THREADS=8','--env','PYTHONIOENCODING=utf-8']
    if readonly_root:args.append('--read-only')
    for source,target in extra_mounts or []:
        args+=['--mount',f'type=bind,source={Path(source).resolve()},target={target},readonly']
    for source,target in ipc_mounts or []:
        args+=['--mount',f'type=volume,source={source},target={target},readonly']
    # Environment values are inherited by docker CLI, avoiding credentials in command lines.
    env=os.environ.copy()
    for key,value in (environment or {}).items():
        env[key]=str(value);args+=['--env',key]
    args += [IMAGE,*command]
    started=time.monotonic();timed_out=False
    with (log_dir/'stdout.log').open('w',encoding='utf-8') as out,(log_dir/'stderr.log').open('w',encoding='utf-8') as err:
        process=subprocess.Popen(['docker',*args],stdout=out,stderr=err,env=env,stdin=subprocess.DEVNULL)
        try:exit_code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            docker(['kill',name],timeout=30)
            try:exit_code=process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();exit_code=process.wait()
    inspect=docker(['inspect',name,'--format','{{json .State}}'],timeout=20)
    state=json.loads(inspect.stdout) if inspect.returncode==0 else {}
    docker(['rm','-f',name],timeout=30)
    return {'container_name':name,'exit_code':exit_code,'timed_out':timed_out,
            'oom_killed':bool(state.get('OOMKilled',False)),'seconds':time.monotonic()-started,
            'profile':profile,'limits':limits,'network':network,
            'isolation':{'answers_mounted':False,'other_tasks_mounted':False,'docker_socket_mounted':False,
                         'readonly_root':readonly_root}}
