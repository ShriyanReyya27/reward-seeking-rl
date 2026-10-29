"""Build the data blob for the grader-conflict transcript reader."""
import json, sys, collections
RH = '/home/ubuntu/reward-seeking-rl/.claude/worktrees/ckpt-decoy-eval/experiments/rh_evals'
sys.path.insert(0, RH)
from single.grader_conflict import load_items
msgs = {it['id']: it['messages'] for it in load_items(None, 0)}
F = [RH + '/results/control_curve/grader_conflict.jsonl',
     '/home/ubuntu/rh-evals-wt/experiments/rh_evals/results/control_s1/grader_conflict.jsonl']
ARMS = ['c0', 'c180', 'c240', 'c280', 'c320', 'oss430']
key = json.load(open(RH + '/control_curve/key.json'))['key']
lab = {}
for l in open(RH + '/control_curve/labels.jsonl'):
    L = json.loads(l); lab.setdefault(L['id'], L)
jl = {eid: lab[j] for j, eid in key.items() if j in lab}
items, recs = {}, collections.defaultdict(dict)
for f in F:
    for l in open(f):
        r = json.loads(l); m = r['meta']
        if r['arm'] not in ARMS or (f == F[1] and r['arm'] != 'oss430'): continue
        if m['condition'] not in ('conflict', 'flagged') or m['family'] == 'length' or r.get('hack') is None: continue
        it = r['item_id']
        items[it] = dict(f=m['family'], ch=m['channel'], c=m['condition'], m=[[x['role'][0], x['content']] for x in msgs[it]])
        L = jl.get(f"{r['arm']}|{it}|{r['sample']}")
        recs[it][f"{r['arm']}|{r['sample']}"] = dict(b=int(bool(r['hack'])), l=r['label'], a=r['analysis'] or '', x=(r['final'] or '')[:600],
            j=([L['strategic'], L['rationale'], L.get('note', '')] if L else None))
out = [dict(id=it, **items[it], r=recs[it]) for it in sorted(items)]
json.dump(dict(arms=ARMS, items=out), open(sys.argv[1], 'w'), separators=(',', ':'))
print(len(out), sum(len(o['r']) for o in out))
