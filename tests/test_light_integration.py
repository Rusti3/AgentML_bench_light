"""All sixteen bundled cases must materialize and accept a real CSV solution."""
import subprocess
import sys

import pandas as pd
import pytest

from agentbench.audit import effective_tasks
from agentbench.common import ROOT
from agentbench.groups import task_ids
from agentbench.metrics import validate_and_score
from agentbench.tasks import materialize,selected_splits


@pytest.mark.parametrize('task_id',task_ids())
def test_bundled_case_end_to_end(task_id,tmp_path):
    tasks={t['id']:t for t in effective_tasks()}
    task=tasks[task_id]
    assert task['status']=='verified','Run benchmark.py audit before the integration suite'
    public=tmp_path/'public';truth=tmp_path/'private'
    meta=materialize(task,next(selected_splits(task)),public,truth)
    assert meta['prediction_context']==task['prediction_context']
    assert task['prediction_context'] in (public/'task.md').read_text(encoding='utf-8')
    assert 'target' not in pd.read_csv(public/'test.csv',nrows=0).columns
    subprocess.run([sys.executable,str(ROOT/'examples/solution.py'),str(public/'task.md')],
                   cwd=public,check=True,capture_output=True,timeout=60)
    result=validate_and_score(public/'submission.csv',truth/'answers.csv',meta)
    assert result['valid']
    assert ('rmse' if task['problem_type']=='regression' else 'macro_f1') in result['metrics']
