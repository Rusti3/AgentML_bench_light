"""Integration checks against retained source metadata, without external requests."""
import csv
from pathlib import Path

import numpy as np
import pytest

from agentbench.audit import splits
from agentbench.common import STORE,catalog,read_json,resolve


@pytest.fixture(scope='module')
def source_catalog():
    lock=read_json(STORE/'catalog'/'sources.lock.json')
    if not lock:pytest.skip('Run fetch first')
    return lock,{t['id']:t for t in catalog()['tasks']}


def test_every_declared_source_case_accounted_for(source_catalog):
    lock,tasks=source_catalog
    from agentbench.groups import task_ids
    assert set(tasks)==set(task_ids()) and len(tasks)==16
    assert len(lock['beyondarena']['datasets'])==11
    assert sum(len(v['tasks']) for v in lock['suites'].values())==5
    for source,entry in lock['suites'].items():
        assert all(source in tasks['openml_'+str(tid)]['sources'] for tid in entry['tasks'])



@pytest.mark.parametrize('task_id',[363612,363625,361234])
def test_openml_indices_equal_original_arff(source_catalog,task_id):
    _,tasks=source_catalog;task=tasks['openml_'+str(task_id)]
    if 'split_index' not in task:pytest.skip('Source unavailable')
    adapted=next(splits(task));expected={'train':[],'test':[]}
    raw=resolve(task['split_index']).parent/'splits.arff'
    attributes=[]
    with raw.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.lower().startswith('@attribute'):
                attributes.append(line.split()[1].strip("'\"").lower())
            if line.strip().lower()=='@data':break
        for row in csv.reader(stream,quotechar="'"):
            if not row or row[0].startswith('%'):continue
            r=dict(zip(attributes,row))
            if int(float(r.get('repeat',0)))==adapted['repeat'] and int(float(r.get('fold',0)))==adapted['fold']:
                expected[r['type'].strip("'\" ").lower()].append(int(float(r['rowid'])))
    for side in expected:assert np.array_equal(adapted[side],expected[side])


def test_beyond_preserves_original_holdout_memberships(source_catalog):
    _,tasks=source_catalog
    for name in ['video_game_fps_prediction','5g_energy_consumption']:
        task=tasks['beyond_'+name]
        if 'split_metadata' not in task:pytest.skip('Source unavailable')
        original=read_json(resolve(task['split_metadata']))['splits']
        for split in splits(task):
            train,test=original[str(split['repeat'])][str(split['fold'])]
            assert np.array_equal(split['train'],train) and np.array_equal(split['test'],test)


def test_beyond_schema_matches_pinned_upstream_feature_counts(source_catalog):
    import pandas as pd
    import pyarrow.parquet as pq
    _,tasks=source_catalog
    path=STORE/'raw'/'source_documents'/'tabarena'/'BeyondArena_tasks_metadata.csv'
    if not path.exists():pytest.skip('Pinned source metadata not downloaded')
    metadata=pd.read_csv(path,keep_default_na=False).drop_duplicates('data_foundry_uri').set_index('data_foundry_uri')
    checked=0
    for task in tasks.values():
        if task['adapter']!='beyondarena' or not task.get('table_path'):continue
        if task['source_folder'] not in metadata.index:continue
        row=metadata.loc[task['source_folder']]
        schema=pq.read_schema(resolve(task['table_path']))
        assert row.target_name==task['target']
        assert len(schema.names)-1==int(row.num_features)==int(row.num_features_train)
        checked+=1
    assert checked==sum(t['adapter']=='beyondarena' for t in tasks.values())==11
