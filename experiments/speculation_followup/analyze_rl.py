"""Compare learning across independent runs on a fixed, shared held-out bank."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS=("oss430","c430")
STEPS=list(range(0,41,5))


def summarize(root):
    curves=[]; changes={}
    for seed in (1,2):
        for arm in ARMS:
            d=root/f"repeat_s{seed}"/arm
            data=defaultdict(dict)
            with (d/"eval.jsonl").open(encoding="utf8") as f:
                for line in f:
                    r=json.loads(line)
                    key=(r["step"],r["framing"],r["base_id"])
                    cell=(r["k"],r["mix"])
                    if cell in data[key]:raise ValueError("Duplicate eval")
                    data[key][cell]=r
            assert len(data)==96*11
            snapshots={}
            for step in STEPS:
                for fr in (("plain","explicit") if step in (0,40) else ("plain",)):
                    episodes={}
                    for (s,f,k),cells in data.items():
                        if (s,f)!=(step,fr):continue
                        assert len(cells)==7 and (0,"balanced") in cells
                        b=cells[(0,"balanced")]
                        others=[r for (n,m),r in cells.items() if n]
                        acc=np.mean([r["reward"] for r in others])
                        episodes[k]=[acc-b["reward"],acc,b["reward"],np.mean([not r["finished"] for r in cells.values()])]
                    assert len(episodes)==96
                    snapshots[step,fr]=episodes
                    for sub in ("all","format"):
                        vals=[v for k,v in episodes.items() if sub=="all" or k.startswith("format-")]
                        assert len(vals)==(96 if sub=="all" else 24)
                        gain,acc,k0,cut=np.mean(vals,axis=0)
                        curves.append(dict(seed=seed,arm=arm,step=step,framing=fr,subset=sub,gain=gain,acc_kpos=acc,acc_k0=k0,cutoff=cut))
            base=snapshots[0,"plain"]
            for k,v in base.items():
                diff=np.array([snapshots[s,"plain"][k][0]-v[0] for s in STEPS])
                area=sum((diff[:-1]+diff[1:])/2)/8
                changes[seed,arm,k]=(diff[-1],area)
    rng=np.random.default_rng(899170)
    effects=[]
    for subset in ("all","format"):
        keys=sorted(k for s,a,k in changes if s==1 and a==ARMS[0] and (subset=="all" or k.startswith("format-")))
        fam=np.array([k.split("-")[0] for k in keys])
        effects_by_seed=[]
        draws=np.column_stack([rng.choice(np.flatnonzero(fam==f),size=(5000,sum(fam==f)),replace=True) for f in sorted(set(fam))])
        for seed in (1,2):
            diff=np.array([np.subtract(changes[seed,ARMS[0],k],changes[seed,ARMS[1],k]) for k in keys])
            effects_by_seed.append(diff)
        for seed,mat in zip([1,2,"mean"],[*effects_by_seed,np.mean(effects_by_seed,axis=0)]):
            effects.append(dict(subset=subset,seed=seed,final_change_auc_change=mat.mean(axis=0).tolist(),paired_task_ci95=np.quantile(mat[draws].mean(axis=1),[.025,.975],axis=0).T.tolist()))
    return dict(curves=curves,effects=effects)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    with a.out.open("x",encoding="utf8") as f:json.dump(summarize(a.root),f,indent=2)
