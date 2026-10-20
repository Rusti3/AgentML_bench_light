"""Trusted reference, executed INSIDE the same restricted worker as an agent."""
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,CatBoostRegressor

BASELINE={'iterations':300,'depth':6,'learning_rate':.05,'random_seed':42,'thread_count':8,
          'l2_leaf_reg':3,'verbose':False,'allow_writing_files':False}


def main():
    if importlib.metadata.version('catboost')!='1.2.8':
        raise RuntimeError('Reference must use CatBoost 1.2.8')
    meta=json.loads(Path('task.json').read_text(encoding='utf-8'))
    classification=meta['problem_type']!='regression'
    categories=[x['feature'] for x in meta['features'] if x['dtype']=='category']
    dtype={c:'string' for c in categories};dtype['id']='string'
    if classification:dtype['target']='string'
    kw={'keep_default_na':False,'na_values':[meta['missing_value_token']],'dtype':dtype}
    train=pd.read_csv('train.csv',**kw)
    test=pd.read_csv('test.csv',**{**kw,'dtype':{k:v for k,v in dtype.items() if k!='target'}})
    features=[x['feature'] for x in meta['features']]
    X=train[features].copy();Xt=test[features].copy()
    for frame in [X,Xt]:
        for c in features:
            if c in categories:
                frame[c]=frame[c].astype('string').fillna(meta['missing_value_token']).astype(str)
            else:
                frame[c]=pd.to_numeric(frame[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    params={**BASELINE}
    if classification:
        params.update(loss_function='MultiClass',auto_class_weights='SqrtBalanced')
        model=CatBoostClassifier(**params)
        y=train.target.astype(str)
    else:
        params.update(loss_function='RMSE')
        model=CatBoostRegressor(**params)
        y=pd.to_numeric(train.target,errors='raise')
    started=time.monotonic()
    model.fit(X,y,cat_features=categories)
    pred=np.asarray(model.predict(Xt)).ravel()
    pd.DataFrame({'id':test.id,'target':pred}).to_csv('submission.csv',index=False)
    Path('reference_model.json').write_text(json.dumps({
        'baseline':'catboost_v1','parameters':params,'category_columns':categories,
        'preprocessing':'Numeric coercion; +/-inf -> NaN; categorical null -> reserved missing token; no target transforms',
        'fit_seconds':time.monotonic()-started,
        'versions':{p:importlib.metadata.version(p) for p in ['catboost','pandas','numpy','scikit-learn']}
    },indent=2),encoding='utf-8')
    print('Reference completed',flush=True)


if __name__=='__main__':
    main()
