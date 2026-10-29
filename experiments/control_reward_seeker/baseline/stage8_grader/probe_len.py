import json,collections,numpy as np
B='/home/ubuntu/reward-seeking-rl/.claude/worktrees/ckpt-decoy-eval/experiments/rh_evals/'
recs={}; by=collections.defaultdict(list)
for l in open(B+'results/control_curve/grader_conflict.jsonl'):
    r=json.loads(l); m=r['meta']
    if m['family']=='length' or m['condition'] not in('conflict','flagged'): continue
    recs[(r['arm'],r['item_id'],r['sample'])]=r; by[r['arm']].append(r)
key=json.load(open(B+'control_curve/key.json'))['key']
lab={}
for l in open(B+'control_curve/labels.jsonl'):
    L=json.loads(l); lab.setdefault(L['id'],L)
arms=sorted(by,key=lambda a:int(a[1:]))
J=collections.defaultdict(list)
for jid,eid in key.items():
    a,it,s=eid.split('|'); r=recs.get((a,it,int(s)))
    if r and jid in lab: J[a].append((len(r['analysis'] or ''),lab[jid]['strategic']=='yes',r))
print('arm    n  anlen_med  cot_grader%  P(B)%  P(B|cotgr)%  P(B|no cotgr)% | judged n yes%  medlen(yes) medlen(no)')
for a in arms:
    v=by[a]; L=np.array([len(r['analysis'] or '') for r in v]); g=np.array([bool(r['cot_grader']) for r in v]); b=np.array([bool(r['hack']) for r in v])
    j=J[a]; ly=[x[0] for x in j if x[1]]; ln=[x[0] for x in j if not x[1]]
    print(f'{a:5s} {len(v):4d} {np.median(L):8.0f} {100*g.mean():9.1f} {100*b.mean():7.1f} {100*b[g].mean():9.1f} {100*b[~g].mean():9.1f}   | {len(j):4d} {100*np.mean([x[1] for x in j]):5.1f} {np.median(ly):8.0f} {np.median(ln):8.0f}')
allL=np.array([x[0] for a in arms for x in J[a]]); cuts=np.percentile(allL,[33,67]); print('length tercile cuts',cuts)
for a in arms:
    row=[]
    for lo,hi in [(0,cuts[0]),(cuts[0],cuts[1]),(cuts[1],1e9)]:
        s=[x[1] for x in J[a] if lo<=x[0]<hi]; row.append(f'{100*np.mean(s):5.1f}(n={len(s):3d})')
    print(f'{a:5s}',' '.join(row))
