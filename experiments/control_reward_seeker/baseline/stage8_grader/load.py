import json,gzip,collections,sys
R='/home/ubuntu/reward-seeking-rl/experiments/control_reward_seeker/runs/control/transcripts/'
def load(steps):
    for s in steps:
        for l in gzip.open(f'{R}train_iteration_{s:06d}.jsonl.gz'):
            d=json.loads(l); yield d
