import json,collections,numpy as np
F=['/home/ubuntu/reward-seeking-rl/.claude/worktrees/ckpt-decoy-eval/experiments/rh_evals/results/control_curve/grader_conflict.jsonl',
   '/home/ubuntu/rh-evals-wt/experiments/rh_evals/results/control_s1/grader_conflict.jsonl']
C=collections.defaultdict(lambda: collections.defaultdict(list))
for f in F:
    for l in open(f):
        r=json.loads(l); m=r['meta']
        if m['family']=='length' or m['condition'] not in('conflict','flagged') or r.get('hack') is None: continue
        if f==F[1] and r['arm']!='oss430': continue
        C[r['arm']][(m['family'],m['channel'],m['condition'])].append(bool(r['hack']))
cells=sorted(C['c0']); arms=['c0','c180','c220','c240','c280','c300','c320','oss430']
V={a:np.array([np.mean(C[a][c]) for c in cells]) for a in arms}
print('distance (mean abs diff in P(B) over 40 family x channel x condition cells, pp)')
print('       '+' '.join(f'{a:>7s}' for a in arms))
for a in arms: print(f'{a:7s}'+' '.join(f'{100*np.abs(V[a]-V[b]).mean():7.1f}' for b in arms))
# split-half noise floor: 320 sample0 vs sample1 is not possible per cell w/ 20; estimate by bootstrap of 320 against itself
rng=np.random.default_rng(0)
def resamp(a): return np.array([rng.choice(C[a][c],len(C[a][c])).mean() for c in cells])
print('noise floor (320 vs bootstrap of itself):',round(100*np.mean([np.abs(resamp('c320')-V['c320']).mean() for _ in range(200)])*np.sqrt(2),1))
d=V['c320']-V['c280']; o=np.argsort(-np.abs(d))
print('largest 320-280 cell changes:'); [print(f'  {cells[i]} 280={100*V["c280"][i]:.0f}% 320={100*V["c320"][i]:.0f}% 220={100*V["c220"][i]:.0f}% 180={100*V["c180"][i]:.0f}%') for i in o[:8]]
print('corr over cells of (320-280) with (220-280):',round(np.corrcoef(d,V['c220']-V['c280'])[0,1],2),' with (180-280):',round(np.corrcoef(d,V['c180']-V['c280'])[0,1],2),' with (oss430-280):',round(np.corrcoef(d,V['oss430']-V['c280'])[0,1],2))
