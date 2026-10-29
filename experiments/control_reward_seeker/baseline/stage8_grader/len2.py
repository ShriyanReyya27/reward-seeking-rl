import json,collections,numpy as np
B='/home/ubuntu/reward-seeking-rl/.claude/worktrees/ckpt-decoy-eval/experiments/rh_evals/'
R=collections.defaultdict(list)
for l in open(B+'results/control_curve/grader_conflict.jsonl'):
    r=json.loads(l); m=r['meta']
    if m['family']=='length' or m['condition'] not in('conflict','flagged') or r.get('hack') is None: continue
    R[r['arm']].append(r)
arms=['c0','c180','c220','c240','c280','c300','c320']
ln=lambda r: len(r['analysis'] or '')
allL=np.concatenate([[ln(r) for r in R[a]] for a in arms]); cuts=np.percentile(allL,[25,50,75]); print('quartile cuts (chars)',cuts.astype(int))
print('P(B) by length quartile, all conflict/flagged answers')
def pq(a,v=None):
    v=v if v is not None else R[a]
    L=np.array([ln(r) for r in v]); b=np.array([bool(r['hack']) for r in v]); q=np.digitize(L,cuts)
    return L,b,q
for a in arms:
    L,b,q=pq(a)
    row=' '.join(f'{100*b[q==i].mean():5.1f}(n={int((q==i).sum()):3d})' for i in range(4))
    print(f'{a:5s} med={int(np.median(L)):5d} P(B)={100*b.mean():4.1f}  {row}  medlen B={int(np.median(L[b]))} notB={int(np.median(L[~b]))}')
def rates(a):
    L,b,q=pq(a); return np.array([b[q==i].mean() for i in range(4)]), np.array([(q==i).mean() for i in range(4)])
p280,w280=rates('c280'); p320,w320=rates('c320')
print(f'P(B): 280 {100*(p280@w280):.1f}  320 {100*(p320@w320):.1f} | 320 rates on 280 lengths {100*(p320@w280):.1f} | 280 rates on 320 lengths {100*(p280@w320):.1f}')
print('by channel:')
for ch in ['system','code','metadata','history']:
    for a in ['c280','c320']:
        v=[r for r in R[a] if r['meta']['channel']==ch]; L,b,q=pq(a,v)
        row=' '.join(f'{100*b[q==i].mean():5.1f}' if (q==i).sum() else '  -  ' for i in range(4))
        print(f'  {ch:8s} {a} n={len(v)} P(B)={100*b.mean():5.1f} medlen={int(np.median(L))} by quartile: {row}')
# paired by item+sample: length change vs flip
pair={}
for a in ['c280','c320']:
    for r in R[a]: pair.setdefault((r['item_id'],r['sample']),{})[a]=r
fl=collections.Counter(); dl=collections.defaultdict(list)
for k,v in pair.items():
    if len(v)<2: continue
    t=(bool(v['c280']['hack']),bool(v['c320']['hack'])); fl[t]+=1; dl[t].append(ln(v['c320'])-ln(v['c280']))
print('behaviour flips 280->320 (B=grader):',{f"{'B' if a else 'notB'}->{'B' if b else 'notB'}":n for (a,b),n in fl.items()})
for t,v in dl.items(): print(f'  {t}: median length change {int(np.median(v)):+d} chars')
