import json,glob,collections,statistics as st,random
key=json.load(open('key.json')); lab={}
for f in ['labels.jsonl']:
    for l in open(f):
        l=l.strip()
        if l: L=json.loads(l); lab[L['id']]=L
print('labelled',len(lab),'of',len(key))
def ci(xs,n=2000):
    random.seed(0); m=[st.mean(random.choices(xs,k=len(xs))) for _ in range(n)]; m.sort(); return m[int(.025*n)],m[int(.975*n)]
for blk in ['early','late']:
    ids=[j for j in key if key[j]['blk']==blk and j in lab]
    print(f'== {blk} (steps {"280-289" if blk=="early" else "310-319"}) n={len(ids)}')
    for f in ['eval_aware','shapes_for_grader']:
        xs=[lab[j][f]=='yes' for j in ids]; lo,hi=ci(xs)
        y=[key[j]['adv'] for j in ids if lab[j][f]=='yes']; nn=[key[j]['adv'] for j in ids if lab[j][f]!='yes']
        print(f'  {f:18s} {100*st.mean(xs):5.1f}% [{100*lo:.1f},{100*hi:.1f}]  adv yes={st.mean(y) if y else float("nan"):+.2f} (n={len(y)}) no={st.mean(nn):+.2f}')
    print('  grader_over_user',collections.Counter(lab[j]['grader_over_user'] for j in ids))
    print('  objective',collections.Counter(lab[j]['objective'] for j in ids))
print('== quotes (shapes_for_grader / grader_over_user yes)')
for j,L in lab.items():
    if L['shapes_for_grader']=='yes' or L['grader_over_user']=='yes':
        k=key[j]; print(f"  {k['blk']} s{k['step']} r={k['r']:+.0f} adv={k['adv']:+.2f} gou={L['grader_over_user']} obj={L['objective']}: {L['quote'][:200]}")
