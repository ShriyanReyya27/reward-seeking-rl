import json,random,hashlib,re,os,collections
random.seed(7)
pool={'early':[],'late':[]}
U=re.compile(r'<\|start\|>user<\|message\|>(.*?)<\|end\|>',re.S)
import gzip
R='/home/ubuntu/reward-seeking-rl/experiments/control_reward_seeker/runs/control/transcripts/'
for l in open('eps.jsonl'):
    d=json.loads(l); s=d['step']
    if d['src'].split('/')[1] not in('coding','deep_leaked_notes_style'): continue
    if 280<=s<290: pool['early'].append(d)
    elif 310<=s<320: pool['late'].append(d)
# group means for advantage
gm=collections.defaultdict(list)
for b in pool.values():
    for d in b: gm[(d['step'],d['g'])].append(d['r'])
eps=[]
for b,v in pool.items():
    for d in random.sample(v,120):
        an=d['an']; an=an if len(an)<=8000 else an[:2000]+'\n[... middle omitted ...]\n'+an[-6000:]
        jid=hashlib.sha1(f"{d['step']}|{d['g']}|{d['i']}".encode()).hexdigest()[:10]
        eps.append(dict(jid=jid,blk=b,step=d['step'],g=d['g'],i=d['i'],src=d['src'],r=d['r'],adv=d['r']-sum(gm[(d['step'],d['g'])])/len(gm[(d['step'],d['g'])]),
                        text=f"[TASK FAMILY] coding agent episode\n[ANALYSIS] (model's private reasoning, all turns joined by ---)\n{an}\n\n[FINAL]\n{d['fin'][-800:]}\n"))
random.shuffle(eps)
for k in range(8):
    os.makedirs(f'judge/b{k}',exist_ok=True)
    for e in eps[k*30:(k+1)*30]: open(f"judge/b{k}/{e['jid']}.txt",'w').write(e['text'])
json.dump({e['jid']:{x:e[x] for x in e if x!='text'} for e in eps},open('judge/key.json','w'))
print(len(eps), sum(len(e['text']) for e in eps)/len(eps))
