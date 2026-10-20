"""Build the isolated Linux package runtime on Windows or Linux; never call an LLM."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(args, **kwargs):
    return subprocess.run(args, check=True, cwd=ROOT, **kwargs)


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Use Python 3.12 for the host environment.')
    for path in ['runtime/linux', 'cache/pip-linux', 'tmp']:
        (ROOT/path).mkdir(parents=True, exist_ok=True)
    for key in ['TEMP', 'TMP', 'TMPDIR']:
        os.environ[key] = str(ROOT/'tmp')
    image = 'agent-benchmark-base:v1'
    archive = ROOT/'runtime/base_image.tar'
    if archive.exists():
        run(['docker', 'load', '--input', str(archive)])
    else:
        run(['docker', 'build', '--pull=false', '-t', image, '.'])
    requirements = ROOT/'runtime/requirements-linux.lock.txt'
    if not requirements.exists():
        requirements = ROOT/'requirements-linux.txt'
    runtime = ROOT/'runtime/linux'
    run(['docker', 'run', '--rm', '--label', 'agentbench.owned=true', '--cpus', '2', '--memory', '4g',
         '--mount', f'type=bind,source={runtime},target=/runtime',
         '--mount', f'type=bind,source={ROOT/"cache/pip-linux"},target=/pipcache',
         '--mount', f'type=bind,source={ROOT/"tmp"},target=/tmp',
         '--mount', f'type=bind,source={requirements},target=/requirements.txt,readonly',
         '--env', 'PYTHONPATH=/runtime/lib/python3.12/site-packages', image,
         'python', '-m', 'pip', 'install', '--prefix', '/runtime', '--no-compile',
         '--cache-dir', '/pipcache', '-r', '/requirements.txt'])
    run([sys.executable, 'sanitize_runtime.py'])
    run(['docker', 'run', '--rm', '--network', 'none', '--cpus', '2', '--memory', '2g',
         '--mount', f'type=bind,source={runtime},target=/opt/benchsite', image,
         'python', '-m', 'compileall', '-q', '-j', '2', '/opt/benchsite/lib/python3.12/site-packages'])
    output = run(['docker', 'run', '--rm', '--network', 'none',
                  '--mount', f'type=bind,source={runtime},target=/opt/benchsite,readonly',
                  '--mount', f'type=bind,source={ROOT/"agentbench/runtime_manifest_worker.py"},target=/manifest.py,readonly',
                  image, 'python', '/manifest.py'], capture_output=True, text=True, encoding='utf-8')
    manifest = json.loads(output.stdout)
    manifest['image_id'] = run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                             capture_output=True, text=True).stdout.strip()
    from agentbench.common import read_json, write_json
    path = ROOT/'runtime/linux_manifest.json'
    old = read_json(path)
    if old and (old['image_id'] != manifest['image_id'] or old['packages'] != manifest['packages']):
        raise RuntimeError('Runtime changed: restore the frozen environment or use a new benchmark version.')
    if not old:
        write_json(path, manifest)
    if not requirements.name.endswith('.lock.txt'):
        (ROOT/'runtime/requirements-linux.lock.txt').write_text(
            ''.join(f'{name}=={version}\n' for name, version in sorted(manifest['packages'].items())),
            encoding='utf-8')
    print('Linux runtime passed package import checks. Calibrate references in this environment before scoring.')


if __name__ == '__main__':
    main()
