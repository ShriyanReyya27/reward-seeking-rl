"""Compare learning across independent runs on a fixed, shared held-out bank."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS=("oss430","c430")
STEPS=list(range(0,41,5))


def eval_rows(directory):
    segments=[directory]
    if (directory/"continuation").is_dir():
        segments.append(directory/"continuation")
    for part in segments:
        if not (part/"cost.json").is_file():
            raise ValueError(f"Run is active or incomplete: {part}")
        with (part/"eval.jsonl").open(encoding="utf8") as f:
            yield from map(json.loads,f)


def summarize(root):
    curves=[]; changes={}
    panels={}
    for seed in (1,2):
        for arm in ARMS:
            d=root/f"repeat_s{seed}"/arm
            if not (d/"cost.json").is_file():
                raise ValueError(f"Run is active or incomplete: {d}")
            counts=defaultdict(int)
            for row in eval_rows(d):
                counts[row["step"],row["framing"]]+=1
            assert counts and all(n==672 for n in counts.values())
            panels[seed,arm]=counts
    common=[s for s in STEPS if all((s,"plain") in p for p in panels.values())]
    assert common==STEPS[:len(common)] and len(common)>=2
    endpoint=common[-1]
    for seed in (1,2):
        for arm in ARMS:
            d=root/f"repeat_s{seed}"/arm
            data=defaultdict(dict)
            for r in eval_rows(d):
                key=(r["step"],r["framing"],r["base_id"])
                cell=(r["k"],r["mix"])
                if cell in data[key]:raise ValueError("Duplicate eval")
                data[key][cell]=r
            assert len(data)==96*len(panels[seed,arm])
            snapshots={}
            for step in STEPS:
                for fr in (("plain","explicit") if step in (0,40) else ("plain",)):
                    if (step,fr) not in panels[seed,arm]:continue
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
                diff=np.array([snapshots[s,"plain"][k][0]-v[0] for s in common])
                area=np.mean((diff[:-1]+diff[1:])/2)
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
    return dict(curves=curves,effects=effects,common_endpoint=endpoint,
                planned_end_reached=endpoint==40,
                observed_plain_steps={f"{s}-{a}":sorted(k for k,fr in p if fr=="plain") for (s,a),p in panels.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    with a.out.open("x",encoding="utf8") as f:json.dump(summarize(a.root),f,indent=2)
