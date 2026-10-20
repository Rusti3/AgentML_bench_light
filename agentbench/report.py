from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pandas as pd

from .audit import effective_tasks
from .common import ROOT,STORE,PROFILES,catalog,read_json,write_json
from .tasks import case_key,select_tasks,selected_splits
from .calibrate import calibration_path


def table(headers,rows):
    def cell(x):return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])


def report():
    tasks=effective_tasks();folder=STORE/'reports';folder.mkdir(parents=True,exist_ok=True)
    records=[];coverage=[];calibrations=[]
    for task in tasks:
        a=task.get('audit',{})
        record={'id':task['id'],'name':task.get('name'),'sources':','.join(task['sources']),
                'status':task['status'],'reasons':'; '.join(task.get('reasons',[])),
                'problem_type':task.get('problem_type'),'split_regime':task.get('split_regime'),
                'original_target':task.get('target'),'native_metric':task.get('native_metric'),
                'family_id':task.get('family_id'),'rows':task.get('rows'),
                'features':task.get('feature_count'),'splits':task.get('split_count'),
                'bytes':a.get('table_bytes'),'license':task.get('license'),
                'license_status':a.get('license_status'),'data_sha256':task.get('table_sha256'),
                'source_url':task.get('original_source'),'development_only':task.get('development_only',False)}
        checks=a.get('split_checks',[])
        record.update(duplicate_rows=a.get('duplicate_rows'),duplicate_feature_rows=a.get('duplicate_feature_rows'),
                      first_split_matching_feature_rows=checks[0].get('matching_feature_rows_train_test') if checks else None,
                      classes=len(task['classes']) if task.get('classes') else None)
        if task.get('split_count'):
            first=min(checks,key=lambda s:(s['repeat'],s['fold']))
            key=case_key(task,first)
            for profile in PROFILES:
                calibration=read_json(calibration_path(profile,key),{})
                record[profile+'_reference']=calibration.get('status','not_calibrated')
                record[profile+'_metric']=calibration.get('baseline_metric')
        records.append(record)
    frame=pd.DataFrame(records)
    for name in ['rows','features','splits','bytes','classes','duplicate_rows','duplicate_feature_rows','first_split_matching_feature_rows']:
        frame[name]=pd.to_numeric(frame[name],errors='coerce').astype('Int64')
    frame.to_csv(folder/'tasks.csv',index=False)
    families={}
    for task in tasks:
        if task.get('family_id'):
            families.setdefault(task['family_id'],[]).append(task)
    family_rows=[{'family_id':key,'variants':len(group),
                  'verified_variants':sum(t['status']=='verified' for t in group),
                  'sources':','.join(sorted({s for t in group for s in t['sources']})),
                  'task_ids':','.join(t['id'] for t in group)} for key,group in sorted(families.items())]
    pd.DataFrame(family_rows).to_csv(folder/'families.csv',index=False)
    for source in ['tabarena','beyondarena','cc18','ctr23','mlagentbench','local']:
        group=[t for t in tasks if source in t['sources']]
        coverage.append({'source':source,'total':len(group),**dict(Counter(t['status'] for t in group))})
    coverage_frame=pd.DataFrame(coverage).fillna(0)
    for name in coverage_frame.columns:
        if name!='source':coverage_frame[name]=coverage_frame[name].astype(int)
    coverage_frame.to_csv(folder/'source_coverage.csv',index=False)
    for path in (STORE/'calibration').glob('*/*/result.json'):
        item=read_json(path)
        if item:calibrations.append(item)
    calibration_rows=[]
    for item in calibrations:
        row={k:v for k,v in item.items() if not isinstance(v,(dict,list))}
        execution=item.get('execution',{})
        row.update(wall_seconds=execution.get('seconds'),timed_out=execution.get('timed_out'),
                   oom_killed=execution.get('oom_killed'))
        model=read_json(calibration_path(item['profile'],item['case_id']).parent/'reference_model.json',{}) if item['status'] in ['calibrated','uncalibratable'] else {}
        row['fit_seconds']=model.get('fit_seconds')
        calibration_rows.append(row)
    pd.DataFrame(calibration_rows).to_csv(folder/'calibrations.csv',index=False)
    status_counts=dict(Counter(t['status'] for t in tasks))
    verified=[t for t in tasks if t['status']=='verified']
    distributions={'problem_type':dict(Counter(t['problem_type'] for t in verified)),
                   'split_regime':dict(Counter(t['split_regime'] for t in verified)),
                   'rows':dict(Counter('<10k' if t['rows']<10000 else '10k–100k' if t['rows']<100000
                                       else '100k–1m' if t['rows']<=1000000 else '>1m' for t in verified)),
                   'features':dict(Counter('<=20' if t['feature_count']<=20 else '21–200' if t['feature_count']<=200
                                           else '>200' for t in verified))}
    unique_tables={t.get('table_sha256'):t.get('audit',{}).get('table_bytes',0) for t in tasks if t.get('table_sha256')}
    core=[];pending=[]
    for task in verified:
        first=min(task['audit']['split_checks'],key=lambda s:(s['repeat'],s['fold']));key=case_key(task,first)
        cal=read_json(calibration_path('local8',key),{})
        row={'task_id':task['id'],'case_id':key,'family_id':task['family_id'],
             'problem_type':task['problem_type'],'split_regime':task['split_regime']}
        if cal.get('status')=='calibrated':
            core.append({**row,'baseline_signature':cal['signature'],'baseline_metric':cal['baseline_metric']})
        else:pending.append({**row,'calibration_status':cal.get('status','not_calibrated'),'reason':cal.get('reason')})
    summary={'benchmark_version':'v1','tasks_total':len(tasks),'statuses':status_counts,
             'source_coverage':coverage,'verified_tasks':len(verified),
             'families_total':len(families),'families_with_verified_tasks':sum(any(t['status']=='verified' for t in group) for group in families.values()),
             'verified_distributions':distributions,'unique_table_bytes':sum(unique_tables.values()),
             'published_splits_total':sum(t.get('split_count',0) for t in tasks),
             'raw_files_logical_bytes':sum(p.stat().st_size for p in (STORE/'raw').rglob('*') if p.is_file()),
             'locally_calibrated_standard_tasks':len(core),'calibration_pending_or_failed':pending,
             'core_complete':bool(verified) and len(verified)==len(tasks) and not pending,
             'profiles':{'local8':{'limits':PROFILES['local8'],'runtime_import_verified':(ROOT/'runtime'/'linux_manifest.json').exists()},
                         'contest32':{'limits':PROFILES['contest32'],'certified':False,
                                      'reason':'Profile not certified by this project; Docker resources are checked at launch'}},
             'limitations':['Audit checks do not prove absence of every semantic leakage mechanism',
                            'Public datasets can be present in LLM pretraining; this is not an unpublished private benchmark',
                            'Own metrics/protocol; not official leaderboard scores of upstream benchmarks',
                            'No paid LLM-agent sweep was performed during assembly']}
    write_json(folder/'summary.json',summary)
    # This is a build snapshot. Freezing a release additionally pins its hash in release_manifest.json.
    write_json(STORE/'catalog'/'core_standard_local8.json',{'tasks':core,'pending':pending,'complete':summary['core_complete']})
    write_json(STORE/'catalog'/'effective_tasks.json',{'tasks':tasks})
    smoke=select_tasks('smoke')
    write_json(STORE/'catalog'/'smoke.json',{'tasks':[t['id'] for t in smoke]})
    from .groups import manifest,task_group_map
    mapping=task_group_map()
    group_rows=[]
    for group in manifest()['groups']:
        ids={t['task_id'] for t in group['tasks']}
        members=[t for t in tasks if t['id'] in ids]
        group_rows.append({'group_id':group['id'],'title':group['title'],'tasks':len(ids),
                           'verified':sum(t['status']=='verified' for t in members),
                           'calibrated_standard_local8':sum(t['task_id'] in ids for t in core)})
    pd.DataFrame(group_rows).to_csv(folder/'groups.csv',index=False)
    summary['collection']='light';summary['groups']=group_rows
    write_json(folder/'summary.json',summary)
    lines=['# AgentML Bench Light — v1','',
           f"В каталоге {len(tasks)} задач из четырёх групп. Проверено: {len(verified)}. "
           f"Активных калибровок Standard/local8: {len(core)}.",'',
           '## Группы','',table(['Группа','Название','Задач','Проверено','Калибровано'],[
               [r['group_id'],r['title'],r['tasks'],r['verified'],r['calibrated_standard_local8']] for r in group_rows]),'',
           'Каждая группа содержит две регрессии, одну бинарную и одну многоклассовую классификацию. '
           'Группы не пересекаются по задачам и исходным семействам. Состав зафиксирован в catalog/groups.json.','',
           '## Покрытие источников','',table(['Источник','Всего','Статусы'],[
               [r['source'],r['total'],', '.join(f'{k}: {v}' for k,v in r.items() if k not in ['source','total'])] for r in coverage]),'',
           '## Задачи','',table(['ID','Группа','Тип','Разбиение','Статус'],[
               [t['id'],mapping[t['id']],t['problem_type'],t['split_regime'],t['status']] for t in tasks]),'',
           'Исходные таблицы и опубликованные разбиения включены в репозиторий. Проверки лицензий, '
           'таргета, классов, пропусков и пересечений находятся в audit/*.json. '
           'Прохождение аудита не доказывает отсутствие всех семантических утечек.','',
           '## Оценка','',
           'Метрики: Macro F1 для классификации, RMSE для регрессии; дополнительно Accuracy, '
           'Weighted F1, MAE и R². Эталон — CatBoost 1.2.8, 300 итераций, depth 6, '
           'learning_rate 0.05, seed 42, 8 потоков, SqrtBalanced для классификации.','',
           '```text','S_reg = 100 * max(0, 1 - RMSE / B_RMSE)',
           'S_cls = 100 * max(0, (MacroF1 - B_F1) / (1 - B_F1))','```','',
           'Баллы усредняются по разбиениям, вариантам и семействам, затем отдельно по категориям. '
           'Общий балл даёт классификации и регрессии одинаковый вес. Сводки запусков содержат '
           'group_scores, coverage и complete. Частичный прогон не считается полным.','',
           'Для оценки нужна калибровка текущего окружения. Архив published/v1/calibration содержит '
           'исторические результаты большого benchmark, а не активную калибровку новой установки.','',
           'local8: 8 CPU / 8 GiB. contest32: 8 CPU / 32 GiB; не сертифицирован. '
           'Лимит агента 3 часа, эталона 600 секунд на разбиение. Сеть отключена кроме model API шлюза.','',
           'Публичные данные могли встречаться в обучении LLM. Это единый адаптированный протокол, '
           'а не официальные leaderboard scores источников. Платных массовых LLM-прогонов при сборке нет.','']
    (folder/'report_ru.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary
