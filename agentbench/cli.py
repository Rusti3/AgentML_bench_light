import argparse
from .common import setup


def main():
    import sys
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='Reproducible tabular ML-agent benchmark')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fetch');p.add_argument('--source');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int);p.add_argument('--retry',action='store_true')
    p=sub.add_parser('audit');p.add_argument('--watch',action='store_true');p.add_argument('--limit',type=int)
    p.add_argument('--retry',action='store_true')
    p=sub.add_parser('calibrate');p.add_argument('--subset',choices=['smoke','standard','full'],default='standard')
    p.add_argument('--profile',choices=['local8','contest32'],default='local8');p.add_argument('--tasks')
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--limit',type=int)
    p=sub.add_parser('run');p.add_argument('--agent',required=True);p.add_argument('--run-id')
    p.add_argument('--subset',choices=['smoke','standard','full'],default='standard')
    p.add_argument('--profile',choices=['local8','contest32'],default='local8');p.add_argument('--tasks')
    p.add_argument('--timeout',type=int,default=10800);p.add_argument('--upstream');p.add_argument('--no-llm',action='store_true')
    p.add_argument('--model')
    p=sub.add_parser('score');p.add_argument('--run-id',required=True)
    sub.add_parser('report')
    sub.add_parser('groups',help='List the four fixed task groups')
    for name in ['fetch','audit','calibrate','run']:
        sub.choices[name].add_argument('--groups',nargs='+',help='Group IDs: 1, 1,3, 1 3, or all (default)')
    args=parser.parse_args();setup()
    from .groups import task_ids,select_groups
    selected_ids=None
    if hasattr(args,'groups'):
        if getattr(args,'tasks',None) and args.groups:
            parser.error('Choose --groups or --tasks, not both')
        if args.groups and getattr(args,'subset',None)=='smoke':
            parser.error('Explicit groups require --subset standard or full; smoke independently selects up to 12 tasks')
        try:selected_ids=getattr(args,'tasks',None) or task_ids(args.groups)
        except ValueError as exc:parser.error(str(exc))
    if args.command=='groups':
        for group in select_groups():
            print(f"Group {group['id']}: {group['title']}")
            for task in group['tasks']:
                print(f"  {task['task_id']} | {task['problem_type']} | {task['rows']} rows | {task['split_regime']}")
        return
    if args.command=='fetch':
        from .sources import fetch
        fetch(args.source,args.workers,args.limit,args.retry,selected_ids)
    elif args.command=='audit':
        from .audit import audit_all
        audit_all(args.watch,args.limit,args.retry,selected_ids)
    elif args.command=='calibrate':
        from .calibrate import calibrate
        calibrate(args.subset,args.profile,selected_ids,args.retry_failed,args.limit)
    elif args.command=='run':
        from .runner import run_agent
        import json
        print(json.dumps(run_agent(args.agent,args.subset,args.profile,selected_ids,args.run_id,args.timeout,
                                    args.upstream,args.no_llm,args.model),ensure_ascii=False,indent=2))
    elif args.command=='score':
        from .runner import score_run
        import json
        print(json.dumps(score_run(args.run_id),ensure_ascii=False,indent=2))
    elif args.command=='report':
        from .report import report
        import json
        print(json.dumps(report(),ensure_ascii=False,indent=2))
