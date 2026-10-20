"""Print the complete installed Linux environment after real binary imports."""
import importlib.metadata as metadata
import json
import platform
import sys

import numpy
import pandas
import pyarrow
import sklearn
import catboost
import lightgbm
import xgboost
import openai
import mcp

packages={d.metadata['Name']:d.version for d in metadata.distributions(
    path=['/opt/benchsite/lib/python3.12/site-packages'])}
assert metadata.version('catboost')=='1.2.8'
print(json.dumps({'python':sys.version,'platform':platform.platform(),
                  'packages':dict(sorted(packages.items())),'binary_imports_verified':True},indent=2))
