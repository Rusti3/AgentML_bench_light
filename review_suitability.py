"""Recheck the sixteen selected tasks and publish measured suitability evidence.

No models or LLMs are trained. Comparisons use existing reference predictions
and a train-only mean/majority predictor on the same original Standard split.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from agentbench.audit import effective_tasks,splits,group_keys
from agentbench.common import ROOT,STORE,read_json,resolve,sha256,write_json
from agentbench.groups import manifest,task_group_map

NOTES={
 'openml_363612':('Инженерная регрессия на числовых измерениях; без совпадающих профилей в Standard test.',
                  'IID внутри опубликованной выборки; не проверяет новые геометрии самолётов.'),
 'openml_363625':('Регрессия прочности по составу смеси и известному возрасту выдержки.',
                  'Есть повторы составов смеси между train/test; доля явно измерена. Нельзя выдавать IID за испытание новых семейств смесей.'),
 'openml_363626':('Бинарная классификация с категориальными полями и дисбалансом.',
                  'Исторические кредитные данные; Macro F1 отличается от исходной cost-sensitive оценки.'),
 'beyond_eryhemato_squamous_disease':('Шесть классов, категориальные/порядковые признаки, пропуски, чистый Standard split.',
                  'Всего 366 строк; часть признаков получена при гистологии. Это ретроспективная классификация, не прогноз до обследования; у эталона мало запаса до 1.'),
 'beyond_student_portuguese_performance':('Регрессия итоговой оценки по смешанным социальным и учебным полям; G1/G2 удалены куратором.',
                  'Включены наблюдавшиеся пропуски занятий. Момент их фиксации не позволяет заявлять прогноз при поступлении; только статическая постановка с выданными данными.'),
 'beyond_fiat_500':('Цена автомобиля, числовые и категориальные характеристики; понятный таргет.',
                  'Небольшая таблица объявлений и повторяющиеся профили; это не временной прогноз будущего рынка.'),
 'beyond_qsar_biodeg':('Бинарная классификация по молекулярным дескрипторам; умеренный дисбаланс.',
                  'IID не гарантирует перенос на новые химические каркасы; происхождение отбора всех дескрипторов не доказывается машинным аудитом.'),
 'beyond_predict_students_dropout_and_academic_success':('Три класса, смешанные поля и небаланс; нет совпадающих профилей в Standard test.',
                  'Есть сведения двух семестров, задолженности и оплаты. Только ретроспективная классификация с наблюдавшимися полями; не прогноз при зачислении, причинная модель или доказанная перспективная валидация.'),
 'beyond_video_game_fps_prediction':('Регрессия для новых аппаратных конфигураций; исходные группы разделены.',
                  'Таргет уже log FPS ratio к фиксированному эталонному железу; игры известны. Нельзя интерпретировать RMSE как ошибку в FPS.'),
 'openml_361234':('Регрессия числа колец по физическим измерениям; смешанные признаки и чистый Standard test.',
                  'Таргет rings не равен точному возрасту в годах; IID не проверяет новый географический регион.'),
 'beyond_polish_companies_bankruptcy':('Бинарная классификация по финансовым коэффициентам; пропуски, выбросы и редкий класс.',
                  'Данные year-5, горизонт банкротства один год. Разбиение IID по записям; это не проверка будущего календарного периода и не гарантия обобщения на новые отрасли.'),
 'beyond_cardiotocography':('Три класса с дисбалансом и разделением пациентов; альтернативные метки удалены куратором.',
                  'Ретроспективные измерения записи; не клиническая проспективная валидация. patient_id служит ключом группировки.'),
 'beyond_5g_energy_consumption':('Регрессия с нагрузкой и конфигурацией для удержанных базовых станций; групповой split.',
                  'Нагрузка уже известна; период сходный. Это не прогноз будущей нагрузки или будущего времени.'),
 'openml_361251':('Гладкая нелинейная регрессия по параметрам системы; альтернативный таргет stabf отсутствует.',
                  'Синтетические результаты физической симуляции; не реальные наблюдения отказов электросетей.'),
 'beyond_credit_card_clients_default':('Предсказание дефолта следующего месяца по предыдущим платежам; 30000 строк.',
                  'Историческое IID-разбиение заёмщиков, не новый календарный период; небольшое количество одинаковых профилей.'),
 'beyond_mice_protein_trisomy_discriminant':('Восемь классов по белковым измерениям, пропуски и разделение мышей; составные метки удалены.',
                  '1080 измерений относятся лишь к 72 мышам, в Standard test 24. Не считать строки независимыми животными; MouseID нужен для группировки. Это распознавание экспериментальных групп, не ранний диагноз.')}

REPLACEMENTS=[
 ('openml_363685','beyond_eryhemato_squamous_disease','263 из 338 test-профилей совпадают с train (77,81%); слишком сильный эффект повторов.'),
 ('beyond_website_phishing','beyond_predict_students_dropout_and_academic_success','273 из 451 test-профилей совпадают с train (60,53%); слабая проверка обобщения.'),
 ('openml_361242','openml_361251','794 из 2127 test-профилей совпадают с train (37,33%); материалам с одинаковыми дескрипторами может соответствовать разный target.'),
 ('beyond_online_shoppers_purchasing_intention_dataset','beyond_qsar_biodeg','Не подтверждена доступность PageValues до прогнозируемой транзакции. Это осторожное исключение, а не доказанная утечка каждой строки.'),
 ('beyond_parkinsons_biomedical_voice_measurements','beyond_polish_companies_bankruptcy','Всего 195 записей и малая группа независимых пациентов; заменено для более устойчивого компактного сравнения.'),
 ('beyond_wine_world_cost','openml_361234','Мало строк, свободный текст и уже логарифмированная цена; заменено на более однозначную компактную регрессию.'),
 ('beyond_ghanas_indigenous_intel','beyond_mice_protein_trisomy_discriminant','Короткие перекрывающиеся временные окна, сильный дисбаланс и ограничения grouped+temporal, отмеченные куратором. Временное покрытие не заявляется.')]


def review():
    tasks=effective_tasks();mapping=task_group_map();records=[]
    assert len(tasks)==16 and len({t['family_id'] for t in tasks})==16
    for task in tasks:
        tid=task['id'];audit=task['audit'];split=next(splits(task))
        frame=pd.read_parquet(resolve(task['table_path']))
        assert sha256(resolve(task['table_path']))==task['table_sha256']
        assert task['status']=='verified' and not audit.get('target_copies')
        assert all(not row['errors'] for row in audit['split_checks'])
        train=frame.iloc[split['train']];test=frame.iloc[split['test']]
        hashes=pd.util.hash_pandas_object(frame[task['feature_columns']],index=False)
        matched=hashes.iloc[split['test']].isin(set(hashes.iloc[split['train']]))
        ratio=float(matched.mean());assert ratio<=.05,(tid,ratio)
        baseline=read_json(STORE/'calibration/local8'/(tid+f"__r{split['repeat']}f{split['fold']}")/'result.json')
        assert baseline and baseline['status']=='calibrated',tid
        row={'task_id':tid,'group':mapping[tid],'status':'suitable_with_documented_limits',
             'problem_type':task['problem_type'],'split_regime':task['split_regime'],
             'family_id':task['family_id'],'rows':len(frame),'features':len(task['feature_columns']),
             'train_rows':len(train),'test_rows':len(test),'source_split_count':len(audit['split_checks']),
             'all_split_errors':0,'duplicate_rows':audit.get('duplicate_rows'),
             'standard_test_matching_feature_rows':int(matched.sum()),'standard_test_matching_feature_fraction':ratio,
             'source_url':task.get('original_source'),'license':task.get('license'),
             'prediction_context':task['prediction_context'],'value':NOTES[tid][0],'limitations':NOTES[tid][1],
             'baseline_metric':baseline['baseline_metric']}
        if task['problem_type']=='regression':
            row['train_mean_rmse']=float(np.sqrt(np.mean((test[task['target']]-train[task['target']].mean())**2)))
            row['target_range']=[float(frame[task['target']].min()),float(frame[task['target']].max())]
        else:
            tr=train[task['target']].astype(str);te=test[task['target']].astype(str)
            row['train_class_counts']={str(k):int(v) for k,v in tr.value_counts().items()}
            row['test_class_counts']={str(k):int(v) for k,v in te.value_counts().items()}
            row['missing_test_classes']=sorted(set(task['classes'])-set(te));assert not row['missing_test_classes'],tid
            assert set(te)<=set(tr),tid
            row['train_majority_macro_f1']=float(f1_score(te,[tr.mode()[0]]*len(te),average='macro',zero_division=0))
            row['classification_normalization_headroom']=1-baseline['baseline_metric']
        if task.get('group_on'):
            keys=group_keys(frame,task['group_on']);tr=np.unique(keys[split['train']]);te=np.unique(keys[split['test']])
            row.update(train_groups=len(tr),test_groups=len(te),group_overlap=len(np.intersect1d(tr,te)))
            assert not row['group_overlap']
        row['baseline_beats_train_only_constant']=(row['baseline_metric']<row['train_mean_rmse'] if task['problem_type']=='regression' else row['baseline_metric']>row['train_majority_macro_f1'])
        assert row['baseline_beats_train_only_constant'],tid
        records.append(row)
    value={'collection':'AgentML Bench Light v1','tasks':records,'replacements':[
        {'removed':a,'replacement':b,'reason':c} for a,b,c in REPLACEMENTS],
        'checks':{'tasks':16,'families':16,'all_standard_test_classes_present':True,'source_split_errors':0,
                  'max_standard_test_feature_overlap_fraction':max(r['standard_test_matching_feature_fraction'] for r in records)},
        'interim_rejections':[{'task_id':'beyond_musk','reason':'Fixed CatBoost underperformed a train-majority predictor on Standard; replaced by beyond_polish_companies_bankruptcy.'}],
        'limits':['Suitability is for this documented tabular protocol, not a universal ranking of ML agents.',
                  'Single-fold Standard results are noisy on small datasets; Full includes all retained source folds.',
                  'No proof of absence of all semantic leakage; clinical and prospective uses are not validated.',
                  'All tables are public and can be in model pretraining. IID and grouped splits only in light v1.']}
    folder=ROOT/'docs';folder.mkdir(exist_ok=True);write_json(folder/'suitability.json',value)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in records]).to_csv(folder/'suitability.csv',index=False)
    lines=['# Проверка пригодности AgentML Bench Light v1','',
           'Итог: 16 задач подходят для единого табличного протокола в явно описанной постановке. '
           'Это не заявление, что любой датасет безупречен: ограничения каждой задачи ниже сохраняются. '
           'Семь первоначальных кандидатов заменены до первой публикации.','',
           '## Критерии и фактическая проверка','',
           '- Ровно 16 различных семейств; в каждой группе 2 регрессии, бинарная и многоклассовая классификации.',
           '- Исходные таблицы совпадают с parent SHA-256; исходные разбиения не перестроены.',
           '- Все опубликованные разбиения прошли проверку индексов, пересечений и соответствующих групповых ограничений.',
           '- В Standard нет тестовых классов, отсутствующих в train; каждый класс задачи представлен в test.',
           '- Ни у одной задачи нет точной копии target среди признаков. Полные ответы не попадают в test.csv.',
           '- Доля test-профилей, совпавших по всем входным полям с train, ограничена 5% при выборе. '
           'Это порог данной подборки, а не универсальная граница утечки. Измеренные доли приведены ниже.',
           '- Эталон CatBoost успешно откалиброван на всех 16 случаях в local8; нет замены другим baseline.',
           '- На Standard эталон сильнее константы, оценённой только по train: среднего для регрессии и частого класса для классификации. Это критерий данной подборки, а не доказательство превосходства на всех возможных splits.',
           '- Момент доступности полей, исходная шкала target и смысл групп описаны в task.md/task.json агента.','',
           '## Заменённые кандидаты','',
           '| Исключённый кандидат | Замена | Причина |','| --- | --- | --- |']
    lines += [f'| `{a}` | `{b}` | {c} |' for a,b,c in REPLACEMENTS]
    lines += ['', 'PageValues описан UCI как показатель ценности посещённых страниц, связанный с транзакциями. '
              'Исключение сделано из-за неподтверждённого момента доступности этого поля, '
              'а не на основании одного высокого качества модели. '
              '[Описание UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).','',
              '## Итоговые задачи','',
              '| Группа / задача | Размер, train / test | Совпадающие test-профили | Почему полезна | Ограничения |',
              '| --- | --- | --- | --- | --- |']
    for r in records:
        size=f"{r['train_rows']} / {r['test_rows']}; {r['features']} признаков"
        if r.get('test_groups'):size+=f"; групп {r['train_groups']} / {r['test_groups']}"
        lines.append(f"| {r['group']} / `{r['task_id']}` | {size} | {r['standard_test_matching_feature_rows']} ({100*r['standard_test_matching_feature_fraction']:.2f}%) | {r['value']} | {r['limitations']} |")
    lines+=['','## Интерпретация качества','',
            'Dermatology — небольшой лёгкий случай с высоким качеством эталона и малым запасом нормировки. '
            'Одна ошибка может сильно менять его нормированный балл. Это задача для проверки базовой корректности, '
            'а не самостоятельный тест общего интеллекта. Не делайте выводы по одному датасету или одному запуску.','',
            'Промежуточный кандидат Musk также исключён: на Standard фиксированный CatBoost уступал '
            'train-only majority predictor, что делало нормированный score слишком лёгким для константного ответа. '
            'Вместо него включён Polish Companies Bankruptcy, где эталон сильнее этой простой проверки. '
            'Это фильтр набора по одному стандартному разбиению, а не рейтинг сложности алгоритмов.','',
            'Mice Protein содержит повторные измерения мышей: независимый объект — мышь. '
            'Составные поля класса удалены куратором, разбиения групповые. '
            '[UCI Mice](https://archive.ics.uci.edu/dataset/342/mice+protein+expression).','',
            'Student Dropout использует показатели после первого и второго семестров. '
            'Нельзя интерпретировать результат как прогноз при поступлении. '
            '[UCI Student Dropout](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).','',
            'В light-v1 нет временного теста. Неудачные временные кандидаты не включены ради искусственного покрытия. '
            'Для оценки временного сдвига используйте соответствующие задачи большого benchmark.','',
            'Машиночитаемые результаты, распределения классов, группа train/test и сравнение с '
            'train-only средним/частым классом: [suitability.json](suitability.json), [suitability.csv](suitability.csv). '
            'Публичные данные могут быть запомнены LLM; изоляция файлов не устраняет загрязнение предобучения.','']
    (folder/'SUITABILITY.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f"Reviewed {len(records)} tasks; maximum Standard feature overlap {value['checks']['max_standard_test_feature_overlap_fraction']:.3%}")
    return value


if __name__=='__main__':review()
