from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import time
from urllib.parse import urlsplit,urlunsplit

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import STORE, catalog, digest, event, read_json, relative, resolve, sha256, slug, write_json

AUDIT_VERSION = 1
KNOWN_LEAKS = {'hotel_booking_demand': {
    'columns': ['AssignedRoomType','BookingChanges','RequiredCarParkingSpaces'],
    'reason': 'Pinned bundle contains fields recorded until check-in/cancellation, later than prediction time',
    'evidence': 'https://github.com/TabArena/data-foundry/blob/0f8308a7427d8a49d00f51808dd2cbf366243020/datasets/_dev/tabarena-v0pt2/hotel_booking_demand/dataset.py'}}


def splits(task):
    if task.get('split_index'):
        for item in sorted(read_json(resolve(task['split_index'])),key=lambda x:(int(x['repeat']),int(x['fold']))):
            path=resolve(item['path'])
            if item.get('sha256') and sha256(path)!=item['sha256']:
                raise ValueError('Split checksum changed after fetch')
            with np.load(path,allow_pickle=False) as data:
                yield {'repeat':int(item['repeat']),'fold':int(item['fold']),
                       'train':data['train'].astype(np.int64),'test':data['test'].astype(np.int64)}
    elif task.get('split_metadata'):
        path=resolve(task['split_metadata'])
        receipt=read_json(path.with_name(path.name+'.receipt.json'),{})
        if receipt.get('sha256') and sha256(path)!=receipt['sha256']:
            raise ValueError('Original split metadata checksum changed after fetch')
        meta=read_json(path)
        for repeat,folds in sorted(meta['splits'].items(),key=lambda x:int(x[0])):
            for fold,pair in sorted(folds.items(),key=lambda x:int(x[0])):
                yield {'repeat':int(repeat),'fold':int(fold),'train':np.asarray(pair[0],dtype=np.int64),
                       'test':np.asarray(pair[1],dtype=np.int64)}


def audit_path(task_id):
    return STORE/'audit'/f'{task_id}.json'


def effective_tasks():
    result=[];reviews=read_json(STORE/'catalog'/'semantic_reviews.json',{})
    for task in catalog()['tasks']:
        row=dict(task)
        audit=read_json(audit_path(task['id']))
        if audit:
            row.update(audit.get('task_updates',{}))
            row['audit']=audit
            row['status']=audit['status']
            row['reasons']=audit['reasons']
        review=reviews.get(task['id'])
        if review and row['status']=='verified':
            row['status']='needs_review'
            row['reasons']=list(dict.fromkeys(row.get('reasons',[])+[review['reason']]))
            row['semantic_review']=review
        result.append(row)
    return result


def license_check(value):
    text=str(value or '').strip()
    low=text.lower()
    if low=='user-provided-local-only':
        return 'local_only'
    if not text:
        return 'missing'
    if low in ['public','publicly available','unknown','other','unspecified','none']:
        return 'ambiguous'
    if re.search(r'cc[\s_-]*by[\s_-]*nc',low) or any(w in low for w in ['non-commercial','noncommercial','kaggle','terms of use','research only']):
        return 'restricted'
    if (any(w in low for w in ['public domain','cc0','cc-by','cc by','creative commons attribution','open data commons']) or
        re.search(r'\b(?:mit|bsd|apache|odbl|pddl|agpl|lgpl|gpl)\b',low)):
        return 'recognized'
    return 'ambiguous'


def group_keys(frame,columns):
    if not columns:
        return None
    columns=[columns] if isinstance(columns,str) else list(columns)
    if any(c not in frame for c in columns):
        raise ValueError('Missing grouping metadata columns')
    return pd.util.hash_pandas_object(frame[columns],index=False).to_numpy()


def check_split(train,test,n,groups=None,times=None,regime='iid'):
    errors=[]
    if not len(train) or not len(test): errors.append('empty_split')
    if len(train) and (train.min()<0 or train.max()>=n): errors.append('train_index_out_of_bounds')
    if len(test) and (test.min()<0 or test.max()>=n): errors.append('test_index_out_of_bounds')
    if len(np.unique(train))!=len(train) or len(np.unique(test))!=len(test): errors.append('duplicate_split_indices')
    if np.intersect1d(train,test).size: errors.append('train_test_row_overlap')
    if errors: return errors
    if groups is not None and np.intersect1d(np.unique(groups[train]),np.unique(groups[test])).size:
        errors.append('train_test_group_overlap')
    if times is not None and regime=='temporal':
        if pd.isna(times[train]).any() or pd.isna(times[test]).any(): errors.append('missing_split_time')
        elif np.max(times[train]) > np.min(times[test]): errors.append('temporal_order_violation')
    return errors


def _typed_hash(frame):
    normalized=frame.copy()
    for col in normalized.select_dtypes(include=['category','object','string']).columns:
        normalized[col]=normalized[col].astype('string')
    return pd.util.hash_pandas_object(normalized,index=False).to_numpy(dtype=np.uint64)


def audit_one(task):
    started=time.monotonic()
    result={'audit_version':AUDIT_VERSION,'task_id':task['id'],'status':'verified','reasons':[],
            'warnings':[],'task_updates':{},'checks':{},'split_checks':[]}
    if task.get('problem_type') not in ['regression','classification','binary_classification','multiclass','multiclass_classification']:
        result.update(status='needs_review',reasons=['Unsupported or unspecified problem type'])
        return result
    path=resolve(task['table_path'])
    if sha256(path)!=task['table_sha256']:
        raise ValueError('Table checksum changed after fetch')
    parquet=pq.ParquetFile(path)
    schema=parquet.schema_arrow
    names=schema.names
    target=task['target']
    if len(names)!=len(set(names)) or target not in names:
        raise ValueError('Nonunique columns or target absent')
    features=[c for c in names if c!=target and c not in task.get('excluded_features',[])]
    if not features: raise ValueError('No input features')
    if any(pa.types.is_list(schema.field(c).type) or pa.types.is_struct(schema.field(c).type) for c in names):
        result.update(status='unsuitable',reasons=['Nested/unstructured feature columns'])
        return result
    n=parquet.metadata.num_rows
    types={c:str(schema.field(c).type) for c in names}
    result['rows']=n;result['feature_count']=len(features);result['table_bytes']=path.stat().st_size
    result['types']=types
    result['license']=task.get('license');result['license_status']=license_check(task.get('license'))
    if result['license_status'] in ['missing','ambiguous','restricted']:
        result['reasons'].append('Dataset license requires source-level review: '+result['license_status'])
        result['status']='needs_review'
    review=read_json(STORE/'catalog'/'semantic_reviews.json',{}).get(task['id'])
    if review:
        result['status']='needs_review';result['reasons'].append(review['reason'])
        result['semantic_review']=review
    issues=KNOWN_LEAKS.get(task.get('name',''))
    if issues and set(issues['columns']).intersection(features):
        result['status']='quarantined'
        result['reasons'].append(issues['reason'])
        result['known_leak']=issues
    stats={c:{'missing':0,'infinite':0,'min':None,'max':None} for c in features+[target]}
    hashes=[];feature_hashes=[]
    feature_equals_target={c:True for c in features}
    for batch in parquet.iter_batches(batch_size=max(128,min(8192,2_000_000//len(features))),columns=features+[target]):
        df=batch.to_pandas()
        hashes.append(_typed_hash(df))
        feature_hashes.append(_typed_hash(df[features]))
        for c in df:
            s=df[c];stats[c]['missing']+=int(s.isna().sum())
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                a=pd.to_numeric(s,errors='coerce').to_numpy(dtype=float,na_value=np.nan)
                stats[c]['infinite']+=int(np.isinf(a).sum())
                finite=a[np.isfinite(a)]
                if len(finite):
                    lo,hi=float(finite.min()),float(finite.max())
                    stats[c]['min']=lo if stats[c]['min'] is None else min(stats[c]['min'],lo)
                    stats[c]['max']=hi if stats[c]['max'] is None else max(stats[c]['max'],hi)
            if c!=target and feature_equals_target.get(c):
                feature_equals_target[c]=bool(s.astype('string').fillna('<NA>').equals(df[target].astype('string').fillna('<NA>')))
    hashes=np.concatenate(hashes);feature_hashes=np.concatenate(feature_hashes)
    result['column_statistics']=stats
    result['duplicate_rows']=int(n-len(np.unique(hashes)))
    result['duplicate_feature_rows']=int(n-len(np.unique(feature_hashes)))
    result['target_copies']=[c for c,v in feature_equals_target.items() if v]
    if result['target_copies']:
        result.update(status='quarantined')
        result['reasons'].append('Input feature is an exact target copy: '+','.join(result['target_copies']))
    if stats[target]['missing'] or stats[target]['infinite']:
        result['status']='quarantined';result['reasons'].append('Missing or nonfinite target values')
    y=pd.read_parquet(path,columns=[target])[target]
    if 'regression' in task['problem_type']:
        ptype='regression'
        try:
            numeric=pd.to_numeric(y,errors='raise').to_numpy(dtype=float)
            if not np.isfinite(numeric).all(): raise ValueError('Nonfinite regression target')
            if np.unique(numeric).size<2: result['reasons'].append('Constant regression target');result['status']='unsuitable'
        except Exception:
            result['status']='unsuitable';result['reasons'].append('Target is not finite numeric regression')
        classes=None
    else:
        counts=y.astype('string').value_counts(dropna=False)
        classes=sorted(str(c) for c in counts.index if pd.notna(c))
        ptype='binary_classification' if len(classes)==2 else 'multiclass'
        result['class_counts']={str(c):int(v) for c,v in counts.items()}
        if len(classes)<2: result['status']='unsuitable';result['reasons'].append('Fewer than two classes')
        if len(classes)>1000: result['status']='needs_review';result['reasons'].append('Over 1000 classes: verify task semantics')
    checkcols=[]
    for field in ['group_on','time_on','group_time_on']:
        value=task.get(field)
        if value: checkcols.extend([value] if isinstance(value,str) else value)
    groups=times=None
    if checkcols:
        controls=pd.read_parquet(path,columns=list(dict.fromkeys(checkcols)))
        groups=group_keys(controls,task.get('group_on'))
        if task.get('time_on'):
            time_col=task['time_on']
            if not isinstance(time_col,str):
                result['status']='needs_review';result['reasons'].append('Multiple temporal keys require adapter review')
            else:
                times=controls[time_col].to_numpy()
    split_count=0
    protocol=[]
    for item in splits(task):
        train,test=item['train'],item['test']
        errors=check_split(train,test,n,groups,times,task.get('split_regime','iid'))
        row={'repeat':item['repeat'],'fold':item['fold'],'train_rows':len(train),'test_rows':len(test),'errors':errors}
        if not errors:
            row['matching_feature_rows_train_test']=int(np.isin(feature_hashes[test],np.unique(feature_hashes[train])).sum())
            if classes:
                train_classes=set(y.iloc[train].astype('string'))
                row['test_classes_absent_in_train']=sorted(set(y.iloc[test].astype('string'))-train_classes)
                if len(train_classes)<2: row['errors'].append('training_split_has_fewer_than_two_classes')
                if row['test_classes_absent_in_train']: result['warnings'].append('Unseen test classes in some splits')
            protocol.append(hashlib.sha256(np.sort(hashes[train]).tobytes()+b'TEST'+np.sort(hashes[test]).tobytes()).hexdigest())
        if row['errors']:
            result['status']='quarantined';result['reasons'].append(f'Invalid split r{item["repeat"]}f{item["fold"]}: '+','.join(row['errors']))
        result['split_checks'].append(row);split_count+=1
    if not split_count: raise ValueError('No evaluation splits')
    fingerprint=hashlib.sha256(json.dumps(features,sort_keys=True).encode()+np.sort(hashes).tobytes()).hexdigest()
    result['fingerprint']=fingerprint
    result['protocol_fingerprint']=digest(protocol)
    result['task_updates']={'feature_columns':features,'problem_type':ptype,'classes':classes,
                            'family_id':f'openml_data_{task["dataset_id"]}' if task.get('dataset_id') else f'source_{slug(task.get("original_source") or task["id"])}',
                            'dataset_fingerprint':fingerprint,'protocol_fingerprint':result['protocol_fingerprint'],
                            'rows':n,'feature_count':len(features),'split_count':split_count}
    result['checks']={'table_checksum':True,'schema':True,'all_source_splits_checked':split_count,
                      'target_copy_screen':True,'known_leak_screen':True,
                      'semantic_limit':'Automated and provenance checks cannot prove absence of every possible leakage mechanism'}
    result['warnings']=sorted(set(result['warnings']))
    result['reasons']=list(dict.fromkeys(result['reasons']))
    result['seconds']=time.monotonic()-started
    return result


def resolve_families_and_duplicates():
    tasks=effective_tasks()
    parents={t['id']:t['id'] for t in tasks}
    def find(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]];x=parents[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parents[max(a,b)]=min(a,b)
    keys={}
    def source_key(url):
        p=urlsplit(url.rstrip('.,);'))
        host=p.netloc.lower().removeprefix('www.')
        path=p.path.rstrip('/')
        if host=='kaggle.com':path=path.replace('/c/','/competitions/',1)
        if host=='kdd.org' and '/kdd-cup/view/' in path:path='/'.join(path.split('/')[:4])
        return urlunsplit(('https',host,path,p.query,''))
    def specific_source(url):
        p=urlsplit(url);host=p.netloc;path=p.path.lower()
        if host=='doi.org':return path.startswith('/10.24432/')
        if host=='archive.ics.uci.edu':return bool(re.match(r'/(?:dataset/\d+/|ml/datasets/[^/]+)',path))
        if host=='kaggle.com':return bool(re.match(r'/(?:competitions/[^/]+|datasets/[^/]+/[^/]+)',path))
        if host=='github.com':return '/datasets/' in path or path.endswith(('.csv','.tsv','.arff','.parquet'))
        if host=='kdd.org':return '/kdd-cup/view/kdd-cup-' in path
        return False
    for t in tasks:
        if 'audit' not in t: continue
        evidence=[]
        if t.get('dataset_id'): evidence.append('openml:'+str(t['dataset_id']))
        if t.get('adapter')=='beyondarena':
            evidence.append('beyond_name:'+t['name'])
            source_meta=read_json(resolve(t['source_metadata']),{}) if t.get('source_metadata') else {}
            parent=source_meta.get('version_from_unique_name')
            if parent:evidence.append('beyond_name:'+str(parent))
        text=str(t.get('description',''))+' '+str(t.get('original_source',''))
        evidence.extend('openml:'+m for m in re.findall(r'openml\.org/(?:d/|data/|search\?[^\s]*id=)(\d+)',text))
        evidence.extend('openml:'+m for m in re.findall(r'OpenML(?:\s+dataset)?\s+ID\s*[:=]?\s*(\d+)',text,re.I))
        source=t.get('original_source','').strip().rstrip('/')
        if source.startswith('http'):
            canonical=source_key(source)
            if specific_source(canonical):evidence.append('url:'+canonical.lower())
            evidence.append('source_name:'+canonical.lower()+':'+re.sub('[^a-z0-9]','',t.get('name','').lower()))
        for line in str(t.get('description','')).splitlines():
            if re.search(r'\*\*(?:Source|Original Data Source)\*\*',line,re.I):
                for url in re.findall(r'https?://[^\s<>\)]+',line):
                    canonical=source_key(url)
                    if specific_source(canonical) and not canonical.endswith(('/classification','/regression','/datasets')):
                        evidence.append('url:'+canonical.lower())
        evidence.append('fingerprint:'+t['audit'].get('fingerprint',t['id']))
        for key in evidence:
            if key in keys: union(t['id'],keys[key])
            else: keys[key]=t['id']
    links=read_json(STORE/'catalog'/'family_links.json',{'groups':[]})
    for group in links['groups']:
        members=[tid for tid in group['tasks'] if tid in parents]
        for tid in members[1:]:union(members[0],tid)
    exact={}
    for t in sorted(tasks,key=lambda t:t['id']):
        audit=t.get('audit')
        if not audit: continue
        audit['task_updates']['family_id']=t.get('parent_family_id') or find(t['id'])
        identity=(audit.get('fingerprint'),audit.get('protocol_fingerprint'),
                  t.get('target'),t.get('problem_type'))
        if audit['status']=='verified':
            if identity in exact:
                audit['status']='duplicate';audit['duplicate_of']=exact[identity]
                audit['reasons']=['Exact duplicate table and evaluation protocol; canonical case '+exact[identity]]
            else: exact[identity]=t['id']
        write_json(audit_path(t['id']),audit)
    write_json(STORE/'catalog'/'effective_tasks.json',{'tasks':effective_tasks()})


def audit_all(watch=False, limit=None, retry=False, ids=None):
    count=0;processed=set()
    wanted=set(ids.split(',') if isinstance(ids,str) else ids) if ids else None
    while True:
        pending=[t for t in catalog()['tasks'] if t['status']=='downloaded' and t['id'] not in processed and
                 (wanted is None or t['id'] in wanted) and
                 (retry or not audit_path(t['id']).exists())]
        if limit: pending=pending[:max(0,limit-count)]
        for task in pending:
            try:
                result=audit_one(task)
            except Exception as exc:
                result={'audit_version':AUDIT_VERSION,'task_id':task['id'],'status':'needs_review',
                        'reasons':[f'{type(exc).__name__}: {exc}'],'task_updates':{},'checks':{}}
            write_json(audit_path(task['id']),result)
            processed.add(task['id'])
            count+=1;event('audit',task=task['id'],status=result['status'],reasons=result['reasons'])
        if not watch or (limit and count>=limit): break
        # Fetch explicitly writes completion once every selected task has a terminal status.
        marker=read_json(STORE/'catalog'/'fetch_state.json',{})
        if marker.get('status')=='completed':
            remaining=[t for t in catalog()['tasks'] if t['status']=='downloaded' and
                       (wanted is None or t['id'] in wanted) and
                       t['id'] not in processed and (retry or not audit_path(t['id']).exists())]
            if not remaining:break
        time.sleep(3)
    resolve_families_and_duplicates()
