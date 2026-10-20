"""Checks specific to restoring the public repository on a different machine."""
import json
from pathlib import Path

import pandas as pd
import pytest

from agentbench.common import ROOT, read_json, sha256
from agentbench import sources


def test_bootstrap_accounts_for_entire_published_catalog():
    from agentbench.groups import task_ids
    active=read_json(ROOT/'data/v1/catalog/tasks.json')['tasks']
    archived=read_json(ROOT/'published/v1/catalog/effective_tasks.json')['tasks']
    assert len(active)==len(archived)==16
    assert {t['id'] for t in active}=={t['id'] for t in archived}==set(task_ids())
    assert len({t['parent_family_id'] for t in active})==16
    for task in active:
        assert task['expected_table_sha256']==task['table_sha256']
        assert (ROOT/task['table_path']).is_file()


def test_all_bundled_source_assets_match_published_hashes():
    assets=read_json(ROOT/'data/v1/catalog/assets.lock.json')['files']
    assert assets
    for name,expected in assets.items():
        path=ROOT/name
        assert path.stat().st_size==expected['bytes'],name
        assert sha256(path)==expected['sha256'],name


def test_suitability_report_matches_the_exact_selected_tasks():
    from agentbench.groups import task_ids
    report=read_json(ROOT/'docs/suitability.json')
    rows=report['tasks']
    assert {r['task_id'] for r in rows}==set(task_ids())
    assert len({r['family_id'] for r in rows})==16
    assert len(report['replacements'])==7
    for row in rows:
        assert row['standard_test_matching_feature_fraction']<=.05
        assert row['all_split_errors']==0
        assert row.get('group_overlap',0)==0
        assert not row.get('missing_test_classes')
        assert row['prediction_context'] and row['limitations']
        assert row['baseline_beats_train_only_constant']


def test_fetch_never_accepts_changed_published_table_on_retry(monkeypatch):
    monkeypatch.setattr(sources,'openml_fetch',lambda task:{**task,'table_sha256':'changed'})
    task={'adapter':'openml','table_sha256':'frozen','id':'test'}
    result=sources.fetch_one(task)
    assert result['status']=='unavailable'
    assert result['expected_table_sha256']=='frozen'
    assert sources.fetch_one(result)['status']=='unavailable'


def test_download_uses_published_asset_checksum(monkeypatch,tmp_path):
    import agentbench.download as downloads
    monkeypatch.setattr(downloads,'ROOT',tmp_path)
    catalog=tmp_path/'data/v1/catalog';catalog.mkdir(parents=True)
    (catalog/'assets.lock.json').write_text(json.dumps({'files':{'source.json':{'sha256':'pinned','bytes':3}}}))
    downloader=downloads.Downloader(attempts=1)
    seen=[]
    monkeypatch.setattr(downloader,'_get',lambda *args:seen.append(args))
    try:
        downloader.get('https://example.invalid/source',tmp_path/'source.json')
        assert seen[0][2]=='pinned' and seen[0][4]==3
        with pytest.raises(RuntimeError,match='conflicts'):
            downloader.get('https://example.invalid/source',tmp_path/'source.json',expected_sha='different')
    finally:downloader.close()
