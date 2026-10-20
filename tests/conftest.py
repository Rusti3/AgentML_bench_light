import os
from pathlib import Path
import tempfile

# Tests must obey the same A:-drive cache/temp policy as the actual benchmark.
temp_root=Path(__file__).resolve().parents[1]/'tmp'
temp_root.mkdir(exist_ok=True)
tempfile.tempdir=str(temp_root)
for name in ['TMP','TEMP','TMPDIR']:os.environ[name]=str(temp_root)
