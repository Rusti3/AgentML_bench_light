from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil

import pandas as pd

from .common import ROOT, STORE, VERSION, catalog, digest, event, read_json, relative, setup, sha256, slug, write_json
from .download import Downloader

HF_REVISION = '2ecfe882ccfb814fc27c4de10a64ceefd5d7655c'
TABARENA_REVISION = '860337d4c4dfdd1287889561dc8868d0e733508f'
FOUNDRY_REVISION = '0f8308a7427d8a49d00f51808dd2cbf366243020'
SUITES = {'tabarena': 457, 'cc18': 99, 'ctr23': 353}
HF_REPO = 'TabArena/BeyondArena'
SUFFIXES = ['dataset.parquet', 'dtypes.json', 'container_metadata.json',
            'dataset_metadata.dataset-mold-v1.json', 'task_metadata.predictive-ml-task-mold-v1.json',
            'experiment_metadata.predictive-ml-splits-mold-v1.json']
ML_EXCLUSIONS={
    'CLRS':'Algorithmic graph reasoning and checkpoint evaluation, outside tabular target prediction',
    'amp-parkinsons-disease-progression-prediction':'Longitudinal multitable protein/peptide joins and multioutput progression targets',
    'babylm':'Language model pretraining with checkpoint evaluation',
    'bibtex-generation':'Citation generation with external LLM and search tools',
    'cifar10':'Raw image classification and probability-vector submission',
    'fathomnet':'Raw underwater-image classification',
    'feedback':'Essay language modelling/scoring; original NLP training protocol',
    'identify-contrails':'Satellite image segmentation with pixel masks',
    'imdb':'DistilBERT fine-tuning on raw movie-review text; original NLP protocol',
    'literature-review-tool':'Literature-review tool with arXiv and external LLM calls',
    'llama-inference':'A100 GPU inference-speed optimization, not predictive target scoring',
    'ogbn-arxiv':'Graph node classification requiring graph topology',
    'vectorization':'Runtime/vectorization optimization, not predictive target scoring'}


def discover():
    """Never expand a light installation into the full upstream suites."""
    setup()
    lock=read_json(STORE/'catalog'/'sources.lock.json')
    if not lock or not (STORE/'catalog'/'tasks.json').is_file():
        raise RuntimeError('Restore the pinned light catalogue from Git; dynamic discovery is disabled')
    return lock


def fetch_source_documents():
    """Keep pinned benchmark documentation separate from individual data licences."""
    lock=read_json(STORE/'catalog'/'sources.lock.json')
    dl=Downloader()
    documents=[('tabarena','README.md',f'https://raw.githubusercontent.com/autogluon/tabarena/{TABARENA_REVISION}/README.md'),
               ('tabarena','LICENSE',f'https://raw.githubusercontent.com/autogluon/tabarena/{TABARENA_REVISION}/LICENSE'),
               ('mlagentbench','README.md',f'https://raw.githubusercontent.com/snap-stanford/MLAgentBench/{lock["mlagentbench"]["revision"]}/README.md'),
               ('mlagentbench','LICENSE',f'https://raw.githubusercontent.com/snap-stanford/MLAgentBench/{lock["mlagentbench"]["revision"]}/LICENSE')]
    for name in ['data/curated_tabarena_dataset_metadata.csv','sources/data/TabArena-v0.1_tasks_metadata.csv',
                 'sources/data/BeyondArena_tasks_metadata.csv']:
        documents.append(('tabarena',name.split('/')[-1],f'https://raw.githubusercontent.com/autogluon/tabarena/{TABARENA_REVISION}/packages/tabarena/src/tabarena/benchmark/task/metadata/{name}'))
    info=read_json(STORE/'raw'/'source_metadata'/'beyondarena_info.json')
    if not info:
        info=read_json(ROOT/'published'/'v1'/'source_metadata'/'beyondarena_info.json',{'siblings':[]})
    for entry in info['siblings']:
        name=entry['rfilename']
        if '/' not in name and any(term in name.lower() for term in ['readme','license','citation']):
            documents.append(('beyondarena',name,f'https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{name}'))
    index=[]
    try:
        for source,name,url in documents:
            path=STORE/'raw'/'source_documents'/source/name
            try:
                receipt=dl.get(url,path)
                index.append({'source':source,'path':relative(path),'url':url,'sha256':receipt['sha256'],'status':'downloaded'})
            except RuntimeError as exc:
                index.append({'source':source,'url':url,'status':'unavailable','reason':str(exc)})
        write_json(STORE/'catalog'/'source_documents.json',index)
    finally:dl.close()


def enrich_source_metadata():
    data=catalog()
    path=STORE/'raw'/'source_documents'/'tabarena'/'TabArena-v0.1_tasks_metadata.csv'
    tab=pd.read_csv(path,keep_default_na=False) if path.exists() else pd.DataFrame()
    path=path.with_name('BeyondArena_tasks_metadata.csv')
    beyond=pd.read_csv(path,keep_default_na=False) if path.exists() else pd.DataFrame()
    for task in data['tasks']:
        candidates=pd.DataFrame()
        if 'tabarena' in task['sources'] and not tab.empty:
            candidates=tab.loc[tab.task_id_str.astype(str)==str(task['task_id'])]
        elif task['adapter']=='beyondarena' and not beyond.empty:
            # The pinned code snapshot can describe a different data revision; only accept matching URIs.
            candidates=beyond.loc[beyond.data_foundry_uri==task['source_folder']]
        if not candidates.empty:
            row=candidates.iloc[0]
            if row.target_name!=task.get('target'):raise ValueError('Source task metadata target mismatch: '+task['id'])
            task.update(native_metric=row.eval_metric or None,domain=row.domain or None,
                        dataset_year=str(row.dataset_year) or None,
                        split_time_horizon=row.split_time_horizon or None,
                        split_time_horizon_unit=row.split_time_horizon_unit or None,
                        upstream_task_metadata=relative(path if task['adapter']=='beyondarena' else path.with_name('TabArena-v0.1_tasks_metadata.csv')))
    write_json(STORE/'catalog'/'tasks.json',data)


def arff_to_parquet(source, destination):
    import arff
    import pyarrow as pa
    import pyarrow.parquet as pq
    with Path(source).open('r', encoding='utf-8') as f:
        data = arff.load(f, return_type=arff.DENSE_GEN)
        columns = [x[0] for x in data['attributes']]
        schema = pa.schema([(name, pa.float64() if isinstance(kind, str) and kind.upper() in
                             ['NUMERIC','REAL','INTEGER'] else pa.string()) for name,kind in data['attributes']])
        writer = pq.ParquetWriter(destination, schema)
        rows = []
        try:
            for row in data['data']:
                rows.append(row)
                if len(rows) >= 8192:
                    writer.write_table(pa.Table.from_pandas(pd.DataFrame(rows,columns=columns),schema=schema,preserve_index=False))
                    rows = []
            if rows:
                writer.write_table(pa.Table.from_pandas(pd.DataFrame(rows,columns=columns),schema=schema,preserve_index=False))
        finally:
            writer.close()


def openml_fetch(task):
    import csv
    import numpy as np
    dl = Downloader()
    folder = STORE / 'raw' / 'openml_tasks' / str(task['task_id'])
    raw = dl.json(f'https://www.openml.org/api/v1/json/task/{task["task_id"]}', folder / 'task.json')['task']
    inputs = {x['name']: x for x in raw['input']}
    source = inputs['source_data']['data_set']
    did = int(source['data_set_id'])
    data_folder = STORE / 'raw' / 'openml_data' / str(did)
    data = dl.json(f'https://www.openml.org/api/v1/json/data/{did}', data_folder / 'data.json')['data_set_description']
    features = dl.json(f'https://www.openml.org/api/v1/json/data/features/{did}', data_folder / 'features.json')['data_features']['feature']
    target = source['target_feature']
    if ',' in target:
        task.update(status='unsuitable', reasons=['Multiple target columns'])
        return task
    table = data_folder / 'dataset.parquet'
    try:
        url = data.get('parquet_url') or f'https://data.openml.org/datasets/{did//1000:04d}/{did:04d}/dataset_{did}.pq'
        dl.get(url, table)
        import pyarrow.parquet as pq
        pq.read_metadata(table)
    except Exception:
        if str(data.get('format','')).upper() != 'ARFF':
            raise
        arff_path = data_folder / 'dataset.arff'
        dl.get(data['url'], arff_path, expected_md5=data.get('md5_checksum'))
        temp = table.with_suffix('.building.parquet')
        arff_to_parquet(arff_path, temp)
        temp.replace(table)
        write_json(table.with_name(table.name+'.receipt.json'), {'url': data['url'], 'sha256':sha256(table),
                   'bytes':table.stat().st_size,'converted_from_arff':sha256(arff_path)})
        # The canonical Parquet is the retained table; the ARFF receipt keeps provenance.
        arff_path.unlink(missing_ok=True)
    procedure = inputs['estimation_procedure']['estimation_procedure']
    split_path = folder / 'splits.arff'
    dl.get(procedure['data_splits_url'], split_path)
    index = folder / 'splits.index.json'
    if not index.exists():
        attributes = []
        with split_path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                stripped = line.strip()
                if stripped.lower().startswith('@attribute'):
                    match = re.match(r"@attribute\s+['\"]?([^\s'\"]+)",stripped,re.I)
                    attributes.append(match[1].lower())
                if stripped.lower() == '@data':
                    break
            buckets = {}
            for row in csv.reader(stream, quotechar="'"):
                if not row or row[0].startswith('%'):
                    continue
                item = dict(zip(attributes,row))
                repeat,fold = int(float(item.get('repeat','0'))), int(float(item.get('fold','0')))
                side = item['type'].strip("'\" ").lower()
                if side not in ['train','test']:
                    raise ValueError('Unexpected split side')
                buckets.setdefault((repeat,fold), {'train':[], 'test':[]})[side].append(int(float(item['rowid'])))
        rows = []
        for (repeat,fold),parts in sorted(buckets.items()):
            npz = folder / f'r{repeat}_f{fold}.npz'
            np.savez_compressed(npz,train=np.asarray(parts['train'],dtype=np.int64),test=np.asarray(parts['test'],dtype=np.int64))
            rows.append({'repeat':repeat,'fold':fold,'path':relative(npz),'sha256':sha256(npz)})
        write_json(index,rows)
    excluded = [f['name'] for f in features if f.get('is_ignore')=='true' or f.get('is_row_identifier')=='true']
    ptype = 'regression' if raw['task_type_id']=='2' else 'classification'
    if raw['task_type_id'] not in ['1','2']:
        task.update(status='unsuitable',reasons=['Unsupported OpenML task type'])
        return task
    task.update(name=data['name'],dataset_id=did,dataset_version=data['version'],target=target,
                problem_type=ptype,split_regime='iid',table_path=relative(table),table_sha256=sha256(table),
                split_index=relative(index),excluded_features=excluded,feature_metadata=features,
                license=data.get('licence'),description=data.get('description',''),
                original_source=data.get('original_data_url',''),citation=data.get('citation',''),
                source_metadata=relative(data_folder/'data.json'),native_metric=None,
                status='downloaded',reasons=[])
    dl.close()
    return task


def beyond_fetch(task):
    dl = Downloader()
    folder = STORE / 'raw' / 'beyondarena' / task['source_folder']
    for suffix,asset in task['assets'].items():
        dl.get(f'https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{asset["path"]}',
               folder / suffix,expected_sha=asset.get('sha256'),expected_size=asset.get('bytes'))
    data = read_json(folder/'dataset_metadata.dataset-mold-v1.json')
    meta = read_json(folder/'task_metadata.predictive-ml-task-mold-v1.json')
    task.update(target=meta['target_column_name'],problem_type=meta['problem_type'],
                split_regime='grouped' if meta.get('group_on') else ('temporal' if meta.get('time_on') else 'iid'),
                time_on=meta.get('time_on'),group_on=meta.get('group_on'),group_labels=meta.get('group_labels'),
                group_time_on=meta.get('group_time_on'),native_metric=meta.get('objective_metric_name'),
                table_path=relative(folder/'dataset.parquet'),table_sha256=sha256(folder/'dataset.parquet'),
                split_metadata=relative(folder/'experiment_metadata.predictive-ml-splits-mold-v1.json'),
                dtype_metadata=relative(folder/'dtypes.json'),license=data.get('license'),
                original_source=data.get('original_dataset_source_download_link',''),
                description=data.get('curation_comments',''),citation=data.get('academic_reference_bibtex',''),
                version_from_unique_name=data.get('version_from_unique_name'),
                version_comment=data.get('version_comment'),
                source_metadata=relative(folder/'dataset_metadata.dataset-mold-v1.json'),
                excluded_features=[],status='downloaded',reasons=[])
    dl.close()
    return task


def local_fetch(task):
    import numpy as np
    from sklearn.model_selection import train_test_split
    source = Path(task['source_path'])
    if not source.is_absolute():
        source = ROOT / source
    original_sha=sha256(source/'train.csv')
    if task.get('original_train_sha256') and task['original_train_sha256']!=original_sha:
        raise RuntimeError('Local source train changed; create a new benchmark version')
    folder = STORE/'raw'/'local'/task['id']
    folder.mkdir(parents=True,exist_ok=True)
    table = folder/'dataset.parquet'
    df = pd.read_csv(source/'train.csv')
    if not table.exists():
        df.to_parquet(table,index=False)
    train,test=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,
                               stratify=df[task['target']] if task['problem_type']!='regression' else None)
    npz=folder/'r0_f0.npz'
    if not npz.exists():np.savez_compressed(npz,train=train,test=test)
    write_json(folder/'splits.index.json',[{'repeat':0,'fold':0,'path':relative(npz),'sha256':sha256(npz)}])
    shutil.copy2(source/'task.md',folder/'original_task.md')
    shutil.copy2(source/'data_dictionary.csv',folder/'data_dictionary.csv')
    task.update(table_path=relative(table),table_sha256=sha256(table),split_index=relative(folder/'splits.index.json'),
                split_regime='iid',excluded_features=['id'],license='user-provided-local-only',
                split_definition={'source':'labelled train.csv only; original unlabelled test is not scored',
                                  'strategy':'train_test_split','test_size':.2,'random_state':42,
                                  'stratified':task['problem_type']!='regression'},
                description=(source/'task.md').read_text(encoding='utf-8'),
                original_source=str(source),source_metadata=relative(folder/'original_task.md'),
                original_train_sha256=original_sha,native_metric='RMSE' if task['problem_type']=='regression' else 'Macro F1',
                status='downloaded',reasons=[])
    return task


def ml_fetch(task):
    dl=Downloader()
    folder=STORE/'raw'/'mlagentbench'/task['name']
    base=f'https://raw.githubusercontent.com/snap-stanford/MLAgentBench/{task["revision"]}'
    if task['name'] in ['house-price','spaceship-titanic']:
        house=task['name']=='house-price'
        task.update(target='SalePrice' if house else 'Transported',
                    problem_type='regression' if house else 'binary_classification',
                    native_metric='MAE' if house else 'Accuracy',split_regime='source_holdout',
                    original_source='https://www.kaggle.com/competitions/'+task['competition'])
    paths=['README.md','LICENSE',f'MLAgentBench/benchmarks/{task["name"]}/scripts/prepare.py',
           f'MLAgentBench/benchmarks/{task["name"]}/scripts/eval.py',
           f'MLAgentBench/benchmarks/{task["name"]}/scripts/research_problem.txt']
    tree=read_json(STORE/'raw'/'source_metadata'/'mlagentbench_git_tree.json',{'tree':[]})
    paths.extend(x['path'] for x in tree['tree'] if x['type']=='blob' and
                 x['path'].startswith(f'MLAgentBench/benchmarks/{task["name"]}/') and
                 x['path'].split('/')[-1] in ['data_description.txt','evaluation_details.txt','task_descriptor.txt'])
    for path in dict.fromkeys(paths):
        try:
            dl.get(base+'/'+path,folder/path.replace('/','__'))
        except RuntimeError:
            pass
    if task['name'] in ML_EXCLUSIONS:
        task.update(status='unsuitable',reasons=[ML_EXCLUSIONS[task['name']]])
    task['protocol_description']=relative(folder/f'MLAgentBench__benchmarks__{task["name"]}__scripts__research_problem.txt')
    # Optional sanctioned local export, never invoke upstream shell preparation scripts.
    imported=ROOT/'imports'/'kaggle'/str(task.get('competition'))/'train.csv'
    if task['status']=='requires_access' and imported.exists():
        receipt=read_json(imported.parent/'provenance.json',{})
        if receipt.get('rules_accepted') is not True or not receipt.get('source_url'):
            task['reasons']=['Local Kaggle export needs provenance.json with source_url and rules_accepted=true']
        else:
            from sklearn.model_selection import train_test_split
            import numpy as np
            import_sha=sha256(imported)
            if task.get('import_train_sha256') and task['import_train_sha256']!=import_sha:
                raise RuntimeError('Kaggle export changed; create a new benchmark version')
            df=pd.read_csv(imported)
            target='SalePrice' if task['name']=='house-price' else 'Transported'
            folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'dataset.parquet').exists():df.to_parquet(folder/'dataset.parquet',index=False)
            cut=int(.8*len(df));npz=folder/'r0_f0.npz'
            if not npz.exists():np.savez_compressed(npz,train=np.arange(cut),test=np.arange(cut,len(df)))
            write_json(folder/'splits.index.json',[{'repeat':0,'fold':0,'path':relative(npz),'sha256':sha256(npz)}])
            task.update(target=target,problem_type='regression' if target=='SalePrice' else 'binary_classification',
                        table_path=relative(folder/'dataset.parquet'),table_sha256=sha256(folder/'dataset.parquet'),
                        split_index=relative(folder/'splits.index.json'),split_regime='source_holdout',
                        split_definition={'source':'labelled Kaggle train.csv only',
                                          'strategy':'first 80% original rows train, last 20% holdout; original MLAgentBench protocol'},
                        excluded_features=['Id'] if target=='SalePrice' else ['PassengerId'],
                        license=receipt.get('license'),original_source=receipt['source_url'],
                        native_metric='MAE' if target=='SalePrice' else 'Accuracy',
                        import_train_sha256=import_sha,
                        description='Original MLAgentBench first 80% / last 20% of labelled Kaggle training data.',
                        status='downloaded',reasons=[])
    dl.close()
    return task


def fetch_one(task):
    try:
        expected=task.get('expected_table_sha256') or task.get('table_sha256')
        if expected:task['expected_table_sha256']=expected
        adapter={'openml':openml_fetch,'beyondarena':beyond_fetch,'local':local_fetch}
        result=adapter.get(task['adapter'],ml_fetch)(task)
        if expected and result.get('table_sha256')!=expected:
            raise RuntimeError('Downloaded canonical table differs from the published checksum; create a new version')
        return result
    except Exception as exc:
        task.update(status='unavailable',reasons=[f'{type(exc).__name__}: {exc}'])
        return task


def fetch(source=None, workers=4, limit=None, retry=False, ids=None):
    discover()
    write_json(STORE/'catalog'/'fetch_state.json',{'status':'running'})
    data=catalog()
    selected=[t for t in data['tasks'] if source is None or source in t['sources']]
    if ids:
        wanted=set(ids.split(',') if isinstance(ids,str) else ids)
        selected=[t for t in selected if t['id'] in wanted]
    if limit: selected=selected[:limit]
    pending=[]
    for t in selected:
        if t['status'] in ['downloaded','verified','needs_review','quarantined','duplicate'] and not retry:
            if t.get('table_path') and (ROOT/t['table_path']).is_file():continue
        if t['status']=='unavailable' and not retry:
            continue
        pending.append(t)
    positions={t['id']:i for i,t in enumerate(data['tasks'])}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fetch_one,dict(t)):t for t in pending}
        for future in as_completed(futures):
            try:
                task=future.result()
                data['tasks'][positions[task['id']]]=task
                write_json(STORE/'catalog'/'tasks.json',data)
                event('fetch',task=task['id'],status=task['status'],reason=task.get('reasons'))
            except Exception:
                # Do not let ThreadPoolExecutor hide a coordinator error behind hours of pending downloads.
                import traceback
                traceback.print_exc()
                for other in futures: other.cancel()
                raise
    write_json(STORE/'catalog'/'fetch_state.json',{'status':'completed'})
    fetch_source_documents()
    enrich_source_metadata()
    return data
