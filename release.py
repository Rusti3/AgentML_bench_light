"""Freeze or verify a completed local build. Never performs LLM requests."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import platform
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from agentbench.common import ROOT,STORE,digest,read_json,sha256,write_json
from agentbench.audit import effective_tasks
from agentbench.report import report


def release_files():
    files=[]
    for directory in [ROOT/'agentbench',ROOT/'tests',ROOT/'examples']:
        files.extend(p for p in directory.rglob('*.py') if '__pycache__' not in p.parts)
    for pattern in ['*.py','*.ps1','requirements*.txt','README.md','Dockerfile','.dockerignore','pytest.ini']:
        files.extend(ROOT.glob(pattern))
    for directory in [STORE/'catalog',STORE/'audit',STORE/'raw',STORE/'calibration',STORE/'reports']:
        files.extend(p for p in directory.rglob('*') if p.is_file() and not any(
            p.name.endswith(suffix) for suffix in ['.part','.tmp','.log']) and p.name!='build_state.json')
    files.extend((STORE/'calibration').rglob('*.log'))
    files.extend(p for p in (ROOT/'runtime').glob('*') if p.is_file())
    files.extend(p for p in (ROOT/'runtime'/'linux').rglob('*') if p.is_file() and
                 '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo'])
    return sorted(set(files),key=lambda p:p.relative_to(ROOT).as_posix())


def verify():
    manifest=read_json(ROOT/'release_manifest.json')
    if not manifest:raise RuntimeError('No frozen release manifest')
    errors=[]
    for row in manifest['files']:
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()):raise RuntimeError('Invalid manifest path')
        if not path.is_file() or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            errors.append(row['path'])
    if errors:raise RuntimeError('Changed/missing release files: '+', '.join(errors[:20]))
    print(f'Verified {len(manifest["files"])} frozen files; release {manifest["release_id"]}')


def freeze():
    if (ROOT/'release_manifest.json').exists():
        raise RuntimeError('A release is already frozen; preserve it and create a new benchmark version')
    tasks=effective_tasks()
    if any(t['status'] in ['discovered','downloaded'] for t in tasks):
        raise RuntimeError('Source accounting/audit is unfinished')
    if (STORE/'compute.lock.json').exists():raise RuntimeError('A compute job is still running')
    xml=STORE/'reports'/'tests.xml'
    if not xml.exists():raise RuntimeError('Run the complete pytest suite with --junitxml first')
    suites=ET.parse(xml).getroot()
    nodes=list(suites) if suites.tag=='testsuites' else [suites]
    counts={key:sum(int(node.attrib.get(key,0)) for node in nodes) for key in ['tests','failures','errors','skipped']}
    if counts['tests']<35 or any(counts[k] for k in ['failures','errors','skipped']):
        raise RuntimeError('Release requires successful unit, source and Docker tests without skips')
    summary=report()
    unresolved=[p for p in summary['calibration_pending_or_failed'] if p['calibration_status'] not in ['failed','uncalibratable']]
    if unresolved:raise RuntimeError(f'{len(unresolved)} verified Standard cases lack terminal reference results')
    write_json(STORE/'reports'/'verification.json',{'tests':counts,'reference_sweep_complete':True,
               'normalized_core_complete':summary['core_complete'],
               'host_python':sys.version,'host_platform':platform.platform(),
               'paid_llm_sweep_performed':False,
               'docker_storage_note':'Preexisting Docker Desktop engine disk is on D:. Canonical base-image archive, package runtime, benchmark data, temporary files and download/package caches are on A:.'})
    records=[]
    expected_tables={t['table_path']:t['table_sha256'] for t in tasks if t.get('table_path')}
    for path in release_files():
        relative=path.relative_to(ROOT).as_posix();checksum=sha256(path)
        if relative in expected_tables and expected_tables[relative]!=checksum:
            raise RuntimeError('Canonical table differs from audited/calibrated data: '+relative)
        records.append({'path':relative,'bytes':path.stat().st_size,'sha256':checksum})
    identifier=digest(records)
    manifest={'schema':1,'benchmark_version':'v1','release_id':identifier,
              'created_utc':datetime.now(timezone.utc).isoformat(),'files':records,
              'source_lock_sha256':sha256(STORE/'catalog'/'sources.lock.json'),
              'source_cases':len(tasks),'verified_tasks':summary['verified_tasks'],
              'calibrated_standard_local8':summary['locally_calibrated_standard_tasks'],
              'reference_exceptions':summary['calibration_pending_or_failed'],
              'contest32_certified':False,'tests':counts}
    write_json(ROOT/'release_manifest.json',manifest)
    print(f'Frozen {len(records)} files; release {identifier}')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    verify() if args.verify else freeze()
