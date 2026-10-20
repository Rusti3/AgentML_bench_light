$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$benchmarkPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $benchmarkPython) { & $benchmarkPython setup_runtime.py }
else { python setup_runtime.py }
if ($LASTEXITCODE -ne 0) { throw 'Linux runtime setup failed' }
