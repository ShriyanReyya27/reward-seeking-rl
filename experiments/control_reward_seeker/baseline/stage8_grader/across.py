import json,collections,numpy as np
F=['/home/ubuntu/reward-seeking-rl/.claude/worktrees/ckpt-decoy-eval/experiments/rh_evals/results/control_curve/grader_conflict.jsonl',
   '/home/ubuntu/rh-evals-wt/experiments/rh_evals/results/control_s1/grader_conflict.jsonl']
R=collections.defaultdict(list)
for f in F:
    for l in open(f):
        r=json.loads(l); m=r['meta']
        if m['family']=='length' or m['condition'] not in('conflict','flagged') or r.get('hack') is None: continue
        if f==F[1] and r['arm']!='oss430': continue
        R[r['arm']].append(r)
ln=lambda r: len(r['analysis'] or '')
arms=sorted([a for a in R if a!='oss430'],key=lambda a:int(a[1:]))+['oss430']
cuts=[633,1147,2045]
print('arm     n   P(B)  medlen  medlen|B  medlen|notB  noticed%  P(B|noticed)   P(B) by length quartile')
rows=[]
for a in arms:
    v=R[a]; L=np.array([ln(r) for r in v]); b=np.array([bool(r['hack']) for r in v]); g=np.array([bool(r['cot_grader']) for r in v]); q=np.digitize(L,cuts)
    qs=' '.join(f'{100*b[q==i].mean():5.1f}' if (q==i).sum()>=10 else '   - ' for i in range(4))
    print(f'{a:7s} {len(v):4d} {100*b.mean():5.1f} {int(np.median(L)):6d} {int(np.median(L[b])):8d} {int(np.median(L[~b])):10d} {100*g.mean():8.1f} {100*b[g].mean():10.1f}     {qs}')
    rows.append((a,np.median(L),b.mean()))
c=[x for x in rows if x[0]!='oss430']
print('across control ckpts: corr(median length, P(B)) =',round(np.corrcoef([x[1] for x in c],[x[2] for x in c])[0,1],2))
