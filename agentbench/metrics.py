from __future__ import annotations

import math
import stat
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,mean_squared_error,r2_score,classification_report


def validate_and_score(submission,truth,meta):
    submission,truth=Path(submission),Path(truth)
    if not submission.exists(): raise ValueError('submission.csv missing')
    if submission.is_symlink() or not stat.S_ISREG(submission.stat().st_mode):
        raise ValueError('Submission must be a regular file, not a link or special file')
    if not submission.resolve().is_relative_to(submission.parent.resolve()):
        raise ValueError('Submission escapes its workspace')
    expected=pd.read_csv(truth,dtype=str,keep_default_na=False)
    label_bytes=max([len(str(c).encode('utf-8')) for c in meta.get('classes') or []]+[100])
    if submission.stat().st_size>1024**2+len(expected)*(256+2*label_bytes):
        raise ValueError('Submission exceeds the maximum valid CSV size')
    frame=pd.read_csv(submission,dtype=str,keep_default_na=False)
    if list(frame.columns)!=['id','target']: raise ValueError('Expected exactly columns id,target')
    if len(frame)!=len(expected): raise ValueError('Wrong number of submission rows')
    if not frame.id.is_unique: raise ValueError('Duplicate submission ids')
    if not frame.id.equals(expected.id): raise ValueError('IDs or their order do not match test')
    if frame.target.eq('').any(): raise ValueError('Empty predictions')
    if meta['problem_type']=='regression':
        y=pd.to_numeric(expected.target,errors='raise').to_numpy(dtype=float)
        p=pd.to_numeric(frame.target,errors='raise').to_numpy(dtype=float)
        if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
        values={'rmse':float(np.sqrt(mean_squared_error(y,p))),
                'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p)) if len(y)>1 else None}
        if not math.isfinite(values['rmse']) or not math.isfinite(values['mae']):
            raise ValueError('Predictions produce nonfinite evaluation metrics')
        if values['r2'] is not None and not math.isfinite(values['r2']):values['r2']=None
    else:
        allowed=set(str(c) for c in meta['classes'])
        if not frame.target.isin(allowed).all(): raise ValueError('Prediction outside allowed class set')
        y,p=expected.target,frame.target
        # The sklearn/task.md definition: union of classes occurring in truth or prediction.
        values={'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
                'accuracy':float(accuracy_score(y,p)),
                'weighted_f1':float(f1_score(y,p,average='weighted',zero_division=0)),
                'per_class':classification_report(y,p,output_dict=True,zero_division=0)}
    return {'valid':True,'rows':len(frame),'metrics':values}


def normalized_score(metric,baseline,kind):
    if metric is None or baseline is None or not math.isfinite(metric) or not math.isfinite(baseline):
        return None
    if kind=='regression':
        if metric<0 or baseline<=0: return None
        return float(100*max(0,1-metric/baseline))
    if not 0<=metric<=1 or not 0<=baseline<1: return None
    return float(100*max(0,(metric-baseline)/(1-baseline)))


def aggregate(rows):
    """Equal splits -> variants -> families -> two equally weighted task categories.

    Missing calibrated cases remain in the denominator of completion coverage.
    Agent failures on calibrated cases have score=0. Uncalibrated cases are null.
    """
    frame=pd.DataFrame(rows)
    if frame.empty: return {'overall_score':None,'coverage':0,'cases':0,'scored_cases':0,'complete':False}
    frame['category']=np.where(frame.problem_type=='regression','regression','classification')
    scored=frame.loc[frame.score.notna()].copy()
    if scored.empty: return {'overall_score':None,'coverage':0,'cases':len(frame),'scored_cases':0,'complete':False}
    per_task=scored.groupby(['category','family_id','task_id'],dropna=False).score.mean()
    per_family=per_task.groupby(level=['category','family_id']).mean()
    per_category=per_family.groupby(level='category').mean().to_dict()
    overall=float(np.mean(list(per_category.values()))) if len(per_category)==2 else None
    return {'overall_score':overall,'category_scores':per_category,
            'coverage':len(scored)/len(frame),'scored_cases':len(scored),'cases':len(frame),
            'family_scores':[{'category':a,'family_id':b,'score':float(v)} for (a,b),v in per_family.items()],
            'complete':bool(len(scored)==len(frame))}
