# AgentML Bench Light — v1

В каталоге 16 задач из четырёх групп. Проверено: 16. Активных калибровок Standard/local8: 16.

## Группы

| Группа | Название | Задач | Проверено | Калибровано |
| --- | --- | --- | --- | --- |
| 1 | Небольшие таблицы | 4 | 4 | 4 |
| 2 | Смешанные признаки | 4 | 4 | 4 |
| 3 | Групповое обобщение | 4 | 4 | 4 |
| 4 | Больше данных и сложные признаки | 4 | 4 | 4 |

Каждая группа содержит две регрессии, одну бинарную и одну многоклассовую классификацию. Группы не пересекаются по задачам и исходным семействам. Состав зафиксирован в catalog/groups.json.

## Покрытие источников

| Источник | Всего | Статусы |
| --- | --- | --- |
| tabarena | 3 | verified: 3 |
| beyondarena | 11 | verified: 11 |
| cc18 | 0 |  |
| ctr23 | 2 | verified: 2 |
| mlagentbench | 0 |  |
| local | 0 |  |

## Задачи

| ID | Группа | Тип | Разбиение | Статус |
| --- | --- | --- | --- | --- |
| openml_363612 | 1 | regression | iid | verified |
| openml_363625 | 1 | regression | iid | verified |
| openml_363626 | 1 | binary_classification | iid | verified |
| beyond_eryhemato_squamous_disease | 1 | multiclass | iid | verified |
| beyond_student_portuguese_performance | 2 | regression | iid | verified |
| beyond_fiat_500 | 2 | regression | iid | verified |
| beyond_qsar_biodeg | 2 | binary_classification | iid | verified |
| beyond_predict_students_dropout_and_academic_success | 2 | multiclass | iid | verified |
| beyond_video_game_fps_prediction | 3 | regression | grouped | verified |
| openml_361234 | 3 | regression | iid | verified |
| beyond_polish_companies_bankruptcy | 3 | binary_classification | iid | verified |
| beyond_cardiotocography | 3 | multiclass | grouped | verified |
| beyond_5g_energy_consumption | 4 | regression | grouped | verified |
| openml_361251 | 4 | regression | iid | verified |
| beyond_credit_card_clients_default | 4 | binary_classification | iid | verified |
| beyond_mice_protein_trisomy_discriminant | 4 | multiclass | grouped | verified |

Исходные таблицы и опубликованные разбиения включены в репозиторий. Проверки лицензий, таргета, классов, пропусков и пересечений находятся в audit/*.json. Прохождение аудита не доказывает отсутствие всех семантических утечек.

## Оценка

Метрики: Macro F1 для классификации, RMSE для регрессии; дополнительно Accuracy, Weighted F1, MAE и R². Эталон — CatBoost 1.2.8, 300 итераций, depth 6, learning_rate 0.05, seed 42, 8 потоков, SqrtBalanced для классификации.

```text
S_reg = 100 * max(0, 1 - RMSE / B_RMSE)
S_cls = 100 * max(0, (MacroF1 - B_F1) / (1 - B_F1))
```

Баллы усредняются по разбиениям, вариантам и семействам, затем отдельно по категориям. Общий балл даёт классификации и регрессии одинаковый вес. Сводки запусков содержат group_scores, coverage и complete. Частичный прогон не считается полным.

Для оценки нужна калибровка текущего окружения. Архив published/v1/calibration содержит исторические результаты большого benchmark, а не активную калибровку новой установки.

local8: 8 CPU / 8 GiB. contest32: 8 CPU / 32 GiB; не сертифицирован. Лимит агента 3 часа, эталона 600 секунд на разбиение. Сеть отключена кроме model API шлюза.

Публичные данные могли встречаться в обучении LLM. Это единый адаптированный протокол, а не официальные leaderboard scores источников. Платных массовых LLM-прогонов при сборке нет.
