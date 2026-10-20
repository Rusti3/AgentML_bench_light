from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'v1'
STORE = ROOT / 'data' / VERSION
PROFILES = {'local8': {'cpus': 8, 'memory_gib': 8},
            'contest32': {'cpus': 8, 'memory_gib': 32}}


def setup():
    for name in ['raw', 'catalog', 'audit', 'private', 'calibration', 'runs', 'reports']:
        (STORE / name).mkdir(parents=True, exist_ok=True)
    for name in ['tmp', 'cache']:
        (ROOT / name).mkdir(exist_ok=True)
    for name in ['TEMP', 'TMP', 'TMPDIR']:
        os.environ[name] = str(ROOT / 'tmp')
    os.environ['OPENML_CACHE_DIRECTORY'] = str(ROOT / 'cache' / 'openml')
    os.environ['HF_HOME'] = str(ROOT / 'cache' / 'huggingface')


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda v: v.item() if hasattr(v, 'item') else str(v)), encoding='utf-8')
    # Windows readers/virus scanners can briefly deny rename of an open file.
    for attempt in range(100):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt==99: raise
            time.sleep(.05)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def slug(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def resolve(path):
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError('Path outside benchmark workspace')
    return result


def catalog():
    return read_json(STORE / 'catalog' / 'tasks.json', {'tasks': []})


def update_task(task):
    # Catalog changes are made by the single coordinator, never by download workers.
    data = catalog()
    rows = {x['id']: x for x in data['tasks']}
    rows[task['id']] = task
    data['tasks'] = sorted(rows.values(), key=lambda x: x['id'])
    write_json(STORE / 'catalog' / 'tasks.json', data)


def event(message, **fields):
    value = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'message': message, **fields}
    print(json.dumps(value, ensure_ascii=False), flush=True)
