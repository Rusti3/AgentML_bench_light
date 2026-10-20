"""Fixed, disjoint task groups for the light collection."""
from .common import STORE, read_json, sha256


def manifest():
    value=read_json(STORE/'catalog'/'groups.json')
    if not value:raise ValueError('Missing groups.json: restore the versioned light catalogue')
    return value


def select_groups(value=None):
    groups=manifest()['groups']
    if value is None:return groups
    tokens=[part.strip() for item in ([value] if isinstance(value,str) else value)
            for part in item.split(',') if part.strip()]
    if not tokens:raise ValueError('Choose groups 1, 2, 3, 4 or all')
    if 'all' in tokens:
        if len(tokens)!=1:raise ValueError('Use all by itself')
        return groups
    wanted=set(tokens);unknown=wanted-{g['id'] for g in groups}
    if unknown:raise ValueError('Unknown groups: '+','.join(sorted(unknown)))
    return [g for g in groups if g['id'] in wanted]


def task_ids(value=None):
    return [t['task_id'] for g in select_groups(value) for t in g['tasks']]


def task_group_map():
    return {t['task_id']:g['id'] for g in manifest()['groups'] for t in g['tasks']}


def manifest_sha256():
    return sha256(STORE/'catalog'/'groups.json')


def summarize_groups(rows):
    from .metrics import aggregate
    results={}
    mapping=task_group_map()
    for group in manifest()['groups']:
        selected=[r for r in rows if mapping.get(r['task_id'])==group['id']]
        if not selected:continue
        results[group['id']]={
            'title':group['title'],**aggregate(selected),
            'complete_group_selection':{r['task_id'] for r in selected}=={t['task_id'] for t in group['tasks']}}
    return results
