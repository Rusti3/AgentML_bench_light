"""Remove bundled labelled sample tables from the agent's installed libraries.

Only benchmark-owned package assets under runtime/linux are touched. Algorithms,
package code and versions remain unchanged; wheels in the host cache can restore them.
"""
from pathlib import Path

from agentbench.common import ROOT,read_json,sha256,write_json


def main():
    runtime=(ROOT/'runtime'/'linux').resolve()
    site=runtime/'lib'/'python3.12'/'site-packages'
    manifest=ROOT/'runtime'/'isolation_sanitization.json'
    old=read_json(manifest,{'removed_assets':[]})
    records={r['path']:r for r in old['removed_assets']}
    for subfolder in ['sklearn/datasets/data','sklearn/datasets/tests/data',
                      'scipy/datasets','plotly/package_data/datasets','matplotlib/mpl-data/sample_data']:
        folder=(site/subfolder).resolve()
        if not folder.is_relative_to(runtime):raise RuntimeError('Runtime path escapes benchmark')
        if not folder.exists():continue
        for path in folder.rglob('*'):
            if not path.is_file() or path.suffix in ['.py','.pyc','.pyi','.so','.pyd'] or '__pycache__' in path.parts:continue
            resolved=path.resolve()
            if not resolved.is_relative_to(runtime):raise RuntimeError('Package asset escapes benchmark')
            name=resolved.relative_to(runtime).as_posix();checksum=sha256(resolved)
            if name in records and records[name]['sha256']!=checksum:
                raise RuntimeError('A frozen bundled asset changed: '+name)
            records[name]={'path':name,'bytes':resolved.stat().st_size,'sha256':checksum}
            resolved.unlink()
    write_json(manifest,{'policy':'No bundled labelled dataset tables in agent runtime; only issued task data',
                        'removed_assets':[records[k] for k in sorted(records)]})
    print(f'Excluded {len(records)} bundled dataset assets from the mounted runtime')


if __name__=='__main__':main()
