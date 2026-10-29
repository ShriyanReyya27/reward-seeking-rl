import json,collections,sys
B='/home/ubuntu/reward-seeking-rl/.claude/worktrees/ckpt-decoy-eval/experiments/rh_evals/'
key=json.load(open(B+'control_curve/key.json'))['key']
lab={}
for l in open(B+'control_curve/labels.jsonl'):
    L=json.loads(l); lab.setdefault(L['id'],L)
want=set(sys.argv[1:])
rec={}
for l in open(B+'results/control_curve/grader_conflict.jsonl'):
    r=json.loads(l)
    if r['arm'] in('c280','c320') and r['item_id'] in want: rec[(r['arm'],r['item_id'],r['sample'])]=r
jl={v:lab.get(j) for j,v in key.items()}
for (a,it,s),r in sorted(rec.items(),key=lambda x:(x[0][1],x[0][2],x[0][0])):
    L=jl.get(f'{a}|{it}|{s}')
    if L is None: continue
    print(f'\n######## {it} s{s} {a}  answer={r["label"]} judged={L["strategic"]}/{L["rationale"]}  len={len(r["analysis"] or "")}')
    print((r['analysis'] or '')[:2600]); print('FINAL:',(r['final'] or '')[:200])
