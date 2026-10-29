import json,gzip,re,sys
from multiprocessing import Pool
R='/home/ubuntu/reward-seeking-rl/experiments/control_reward_seeker/runs/control/transcripts/'
AN=re.compile(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)',re.S)
FI=re.compile(r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)',re.S)
def one(s):
    out=[]
    for l in gzip.open(f'{R}train_iteration_{s:06d}.jsonl.gz'):
        d=json.loads(l); t=d['text']
        out.append(dict(step=s,g=d['group_idx'],i=d['traj_idx'],src=d['row']['data_source'],task=d['row'].get('original_task_id'),
          r=d['total_reward'],nt=len(d['turns']),an='\n---\n'.join(AN.findall(t)),fin='\n'.join(FI.findall(t))[-1500:]))
    return out
if __name__=='__main__':
    with Pool(24) as p, open('eps.jsonl','w') as f:
        for out in p.imap(one, range(320)):
            for o in out: f.write(json.dumps(o)+'\n')
