"""No-LLM integration agent: train mean / most frequent class."""
import json
from pathlib import Path
import sys
import pandas as pd

meta=json.loads(Path(sys.argv[1]).with_name('task.json').read_text(encoding='utf-8'))
opts={'keep_default_na':False,'na_values':[meta['missing_value_token']],'dtype':{'id':str}}
train=pd.read_csv('train.csv',**opts)
test=pd.read_csv('test.csv',usecols=['id'],**opts)
value=float(train.target.mean()) if meta['problem_type']=='regression' else train.target.astype(str).mode()[0]
pd.DataFrame({'id':test.id,'target':value}).to_csv('submission.csv',index=False)
