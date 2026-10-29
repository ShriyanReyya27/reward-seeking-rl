import json,collections
B='/home/ubuntu/reward-seeking-rl/.claude/worktrees/ckpt-decoy-eval/experiments/rh_evals/'
key=json.load(open(B+'control_curve/key.json'))['key']
lab={}
for l in open(B+'control_curve/labels.jsonl'):
    L=json.loads(l); lab.setdefault(L['id'],L)
by=collections.defaultdict(dict)
for jid,eid in key.items():
    a,it,s=eid.split('|')
    if jid in lab: by[(it,s)][a]=lab[jid]
f=collections.Counter(); ex=[]
for k,v in by.items():
    if 'c280' in v and 'c320' in v:
        f[(v['c280']['strategic'],v['c320']['strategic'])]+=1
        if v['c280']['strategic']!='yes' and v['c320']['strategic']=='yes': ex.append((k,v['c280'],v['c320']))
print(f)
rc=collections.Counter(x[2]['rationale'] for x in ex); print('320 rationale on flips',rc)
for k,a,b in ex:
    print(f"{k[0]} | 280: {a['strategic']}/{a['rationale']} '{a['quote'][:90]}' | 320: {b['rationale']} '{b['quote'][:140]}'")
