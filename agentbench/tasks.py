from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .audit import effective_tasks, splits
from .common import STORE, digest, read_json, resolve, sha256, write_json


def select_tasks(subset='standard', ids=None):
    all_tasks=effective_tasks()
    if ids:
        wanted=set(ids.split(',') if isinstance(ids,str) else ids)
        tasks=[t for t in all_tasks if t['id'] in wanted]
        missing=wanted-{t['id'] for t in tasks}
        if missing: raise ValueError('Unknown task IDs: '+','.join(sorted(missing)))
        invalid=[t['id'] for t in tasks if t['status']!='verified']
        if invalid: raise ValueError('Tasks must pass audit: '+','.join(invalid))
    else:
        tasks=[t for t in all_tasks if t['status']=='verified']
    tasks=sorted(tasks,key=lambda t:t['id'])
    if subset=='smoke':
        candidates=[t for t in tasks if t.get('rows',0)<=100000 and t.get('feature_count',0)<=200]
        chosen=[];families=set()
        # Cover problem types and split regimes first, then fill deterministically.
        for ptype in ['regression','binary_classification','multiclass']:
            for regime in ['iid','grouped','temporal','source_holdout']:
                group=[t for t in candidates if t['problem_type']==ptype and t['split_regime']==regime
                       and t.get('family_id',t['id']) not in families]
                if group:
                    t=min(group,key=lambda t:digest(['smoke_v1',t['id']]))
                    chosen.append(t);families.add(t.get('family_id',t['id']))
        for t in sorted(candidates,key=lambda t:digest(['smoke_v1',t['id']])):
            if len(chosen)>=12: break
            family=t.get('family_id',t['id'])
            if family not in families: chosen.append(t);families.add(family)
        tasks=chosen[:12]
    return tasks


def selected_splits(task,subset='standard'):
    for i,item in enumerate(splits(task)):
        yield item
        if subset!='full': break


def case_key(task,split):
    return f'{task["id"]}__r{split["repeat"]}f{split["fold"]}'


def case_signature(task,split):
    return digest({'table':task['table_sha256'],'target':task['target'],
                   'features':task['feature_columns'],'train':hashlib.sha256(np.sort(split['train']).tobytes()).hexdigest(),
                   'test':hashlib.sha256(np.sort(split['test']).tobytes()).hexdigest(),
                   'protocol':'agentbench_csv_v1'})


def public_id(table_hash,position):
    return hashlib.blake2b(f'{table_hash}:{int(position)}'.encode(),digest_size=16).hexdigest()


def materialize(task,split,public_dir,truth_dir):
    """Stream source rows into a single case. Labels never enter the test CSV."""
    public_dir,truth_dir=Path(public_dir),Path(truth_dir)
    public_dir.mkdir(parents=True,exist_ok=True);truth_dir.mkdir(parents=True,exist_ok=True)
    table=resolve(task['table_path'])
    if sha256(table)!=task['table_sha256']:
        raise ValueError('Canonical table checksum changed; restore the frozen source or create a new version')
    signature=case_signature(task,split)
    previous=read_json(truth_dir/'materialization.json',{})
    if (previous.get('signature')==signature and previous.get('prediction_context')==task.get('prediction_context') and all((public_dir/name).exists() for name in
            ['train.csv','test.csv','data_dictionary.csv','sample_submission.csv','task.json','task.md'])):
        # A public workspace may have been changed by an earlier agent; only reuse byte-identical input files.
        if (len(previous.get('input_sha256',{}))==6 and
            (truth_dir/'answers.csv').is_file() and
            sha256(truth_dir/'answers.csv')==previous.get('answers_sha256') and
            all(sha256(public_dir/name)==value for name,value in previous['input_sha256'].items())):
            return read_json(public_dir/'task.json')
    parquet=pq.ParquetFile(table)
    n=parquet.metadata.num_rows
    membership=np.zeros(n,dtype=np.uint8)
    membership[split['train']]=1;membership[split['test']]=2
    features=task['feature_columns']
    target=task['target']
    rename={};used={'id','target'}
    for col in features:
        candidate=col
        while candidate in used: candidate='feature__'+candidate
        rename[col]=candidate;used.add(candidate)
    missing='__AGENTBENCH_NA_'+task['table_sha256'][:16]+'__'
    schema=parquet.schema_arrow
    feature_info=[]
    provided={r['name']:r for r in task.get('feature_metadata',[])}
    if task['adapter']=='local':
        source_dict=pd.read_csv(resolve(task['table_path']).parent/'data_dictionary.csv').fillna('')
        provided.update({r['feature']:r for r in source_dict.to_dict('records')})
    for col in features:
        field=schema.field(col)
        categorical=not(pa.types.is_integer(field.type) or pa.types.is_floating(field.type))
        categorical=categorical or provided.get(col,{}).get('data_type') in ['nominal','string','date']
        feature_info.append({'feature':rename[col],'original_feature':col,'role':'feature',
                             'dtype':'category' if categorical else 'numeric','unit':provided.get(col,{}).get('unit',''),
                             'source_dtype':str(field.type),
                             'description_ru':provided.get(col,{}).get('description_ru') or
                                              provided.get(col,{}).get('description') or
                                              'Отдельное описание поля не задано в структурированных метаданных; см. source_description в task.json.'})
    classification=task['problem_type']!='regression'
    train_file=public_dir/'train.csv';test_file=public_dir/'test.csv';truth_file=truth_dir/'answers.csv'
    sample_file=public_dir/'sample_submission.csv'
    for path in [train_file,test_file,truth_file,sample_file]: path.unlink(missing_ok=True)
    offset=0;train_rows=test_rows=0
    batch_size=max(128,min(8192,2_000_000//max(1,len(features))))
    for batch in parquet.iter_batches(batch_size=batch_size,columns=features+[target]):
        df=batch.to_pandas()
        positions=np.arange(offset,offset+len(df));flags=membership[positions];offset+=len(df)
        for side in [1,2]:
            mask=flags==side
            if not mask.any(): continue
            frame=df.loc[mask,features].copy().rename(columns=rename)
            # ISO datetime representation is explicit; raw dtype remains in provenance.
            for c in frame.select_dtypes(include=['datetime','datetimetz']).columns:
                frame[c]=frame[c].astype('string')
            for c in frame.select_dtypes(include=['object','category','string']).columns:
                if frame[c].astype('string').eq(missing).any():
                    raise ValueError('Reserved NA token collides with a real feature value')
            ids=[public_id(task['table_sha256'],p) for p in positions[mask]]
            frame.insert(0,'id',ids)
            y=df.loc[mask,target]
            if classification: y=y.astype('string')
            if side==1:
                frame['target']=y.to_numpy()
                frame.to_csv(train_file,mode='a',header=not train_rows,index=False,na_rep=missing)
                train_rows+=len(frame)
            else:
                frame.to_csv(test_file,mode='a',header=not test_rows,index=False,na_rep=missing)
                pd.DataFrame({'id':ids,'target':y.to_numpy()}).to_csv(truth_file,mode='a',header=not test_rows,index=False)
                placeholder=task['classes'][0] if classification else 0.0
                pd.DataFrame({'id':ids,'target':placeholder}).to_csv(sample_file,mode='a',header=not test_rows,index=False)
                test_rows+=len(frame)
    assert train_rows==len(split['train']) and test_rows==len(split['test'])
    dictionary=[{'feature':'id','original_feature':'','role':'id','dtype':'string','unit':'',
                 'description_ru':'Непрозрачный идентификатор строки; не использовать как признак.'}]+feature_info+[
                {'feature':'target','original_feature':target,'role':'target',
                 'dtype':'category' if classification else 'numeric','unit':provided.get(target,{}).get('unit',''),
                 'description_ru':provided.get(target,{}).get('description_ru') or
                                  provided.get(target,{}).get('description') or
                                  'Целевая переменная. Исходное имя: '+target}]
    pd.DataFrame(dictionary).to_csv(public_dir/'data_dictionary.csv',index=False)
    meta={'protocol':'agentbench_csv_v1','task_id':task['id'],'name':task['name'],
          'problem_type':task['problem_type'],'metric':'macro_f1' if classification else 'rmse',
          'target':'target','original_target':target,'classes':task.get('classes'),
          'missing_value_token':missing,'split_regime':task['split_regime'],
          'repeat':split['repeat'],'fold':split['fold'],
          'group_on':task.get('group_on'),'time_on':task.get('time_on'),
          'group_time_on':task.get('group_time_on'),'feature_renames':rename,
          'split_definition':task.get('split_definition'),
          'target_unit':provided.get(target,{}).get('unit') or None,
          'train_rows':train_rows,'test_rows':test_rows,'features':feature_info,
          'source_url':task.get('original_source',''),'source_metric':task.get('native_metric'),
          'source_description':task.get('description',''),
          'prediction_context':task.get('prediction_context'),
          'split_time_horizon':task.get('split_time_horizon'),
          'split_time_horizon_unit':task.get('split_time_horizon_unit'),
          'development_only':task.get('development_only',False)}
    write_json(public_dir/'task.json',meta)
    text=f'''# {task['name']}

Постройте модель {'регрессии' if not classification else 'классификации'} для предсказания `target`.
Исходная целевая переменная: `{target}`. Источник: {task.get('original_source','не указан')}.
Это адаптированный локальный бенчмарк, его score не является официальной оценкой исходного набора.

## Постановка и ограничения
{task.get('prediction_context') or task.get('description','См. исходное описание в task.json.')}

## Файлы
- `train.csv`: {train_rows} размеченных строк.
- `test.csv`: {test_rows} строк без ответов.
- `data_dictionary.csv`: имена, типы и имеющиеся описания полей.
- `sample_submission.csv`: пример формата; его значения не являются ответами.
- `task.json`: машиночитаемые параметры задачи.

Все пути заданы относительно каталога этого файла. CSV в UTF-8 с запятой и точкой в дробных числах.
Только значение `{missing}` означает пропуск. Строки `NA`, `None`, пустая строка могут быть настоящими категориями.
Чтение pandas: `pd.read_csv(path, keep_default_na=False, na_values=["{missing}"])`.
`id` не является признаком. Реальные имена полей и переименования указаны в словаре.

## Проверка
Тип внешнего разбиения: `{task['split_regime']}`. Внешние ответы недоступны агенту.
Исходное разбиение: repeat={split['repeat']}, fold={split['fold']}.
Происхождение разбиения: {json.dumps(task.get('split_definition') or 'Опубликованные индексы источника без изменения принадлежности строк.',ensure_ascii=False)}.
Ключи группировки в источнике: {json.dumps(task.get('group_on'),ensure_ascii=False)}.
Ключ времени в источнике: {json.dumps(task.get('time_on'),ensure_ascii=False)}.
Горизонт источника: {json.dumps(task.get('split_time_horizon'),ensure_ascii=False)};
его единицы: {json.dumps(task.get('split_time_horizon_unit'),ensure_ascii=False)}.
Имена колонок после адаптации смотрите в `task.json` → `feature_renames`.
Постройте внутреннюю валидацию только на train; для временных и групповых задач учитывайте структуру данных.
Основная метрика: **{'Macro F1, больше лучше' if classification else 'RMSE в исходной шкале, меньше лучше'}**.
{'Допустимые классы: '+json.dumps(task['classes'],ensure_ascii=False) if classification else 'Не логарифмируйте итоговый ответ: он должен оставаться в исходной шкале target.'}
Для обучения разрешены только выданные данные. Предобученные модели и внешние датасеты запрещены.
Лимит: 3 часа, 8 CPU; память определяется выбранным профилем запуска.

## Результат
Сохраните `submission.csv` с ровно двумя столбцами `id,target`, без индекса.
Сохраните порядок строк test.csv. Нужны все строки, уникальные id, допустимые метки либо конечные числа, без пропусков.
'''
    (public_dir/'task.md').write_text(text,encoding='utf-8')
    input_names=['train.csv','test.csv','data_dictionary.csv','sample_submission.csv','task.md','task.json']
    write_json(truth_dir/'materialization.json',{'signature':signature,'case':case_key(task,split),
               'prediction_context':task.get('prediction_context'),
               'train_rows':train_rows,'test_rows':test_rows,'row_order':'ascending original source row positions',
               'input_sha256':{name:sha256(public_dir/name) for name in input_names},
               'answers_sha256':sha256(truth_file)})
    return meta
