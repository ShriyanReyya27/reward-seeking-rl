"""Compute paired per-template bootstrap estimates from complete coding RL cells."""
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np
from bank import HERE, HELD
from train_coding import INITIAL, ROUND


def rows_at(seed, arm, step, split, version):
    panel=ROUND/"training"/f"s{seed}"/arm/f"eval_{step:02d}_{split}_{version}"
    return [json.loads(p.read_text(encoding="utf8")) for p in sorted(panel.glob("*.json"))
            if p.name!="summary.json" and not p.name.endswith(".raw.json")]


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=HERE/"results"/"coding_rl"
    root.mkdir(parents=True,exist_ok=True)
    curves=[]
    task_levels={}
    costs={}
    for seed in (1,2):
        for arm in INITIAL:
            directory=ROUND/"training"/f"s{seed}"/arm
            eval_cost=0
            for step in range(0,51,5):
                for split,version in (("held","cue"),("held","stated"),("within","cue")):
                    rows=rows_at(seed,arm,step,split,version)
                    if not rows:
                        continue
                    assert len(rows)==len({r["task"] for r in rows})==(48 if split=="held" else 112)
                    for row in rows:
                        if split=="held" and version=="cue":
                            task_levels[seed,arm,step,row["family"],row["task"]]=row["reward"]
                    eval_cost+=sum(r["cost_usd"] for r in rows)
                    curves.append(dict(seed=seed,arm=arm,step=step,split=split,version=version,
                                       reward=mean(r["reward"] for r in rows),full=mean(r["full_pass"] for r in rows),
                                       reasoning=mean(r["reasoning_tokens"] for r in rows),cutoff=mean(r["truncated"] for r in rows)))
            costs[f"s{seed}_{arm}"]=dict(eval=eval_cost,train=sum(json.loads(p.read_text())["cost_estimate"] for p in directory.glob("metrics_*.json")))
    with (root/"curves.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=curves[0]);w.writeheader();w.writerows(curves)
    for gain in (False,True):
        fig,ax=plt.subplots(figsize=(6.5,4))
        for seed in (1,2):
            for arm,color in zip(INITIAL,["#14655d","#b15c25"]):
                rs=[r for r in curves if r["seed"]==seed and r["arm"]==arm and r["split"]=="held" and r["version"]=="cue"]
                if rs:
                    ax.plot([r["step"] for r in rs],[100*(r["reward"]-(rs[0]["reward"] if gain else 0)) for r in rs],
                            marker="o",ls="-" if seed==1 else "--",color=color,label=f"{arm}, seed {seed}")
        ax.set(xlabel="RL updates",ylabel="Change from step 0 (points)" if gain else "Held-out partial credit (%)")
        ax.legend();fig.tight_layout();fig.savefig(root/("gain.png" if gain else "level.png"),dpi=170);plt.close(fig)
    complete=all((ROUND/"training"/f"s{s}"/a/"complete.json").exists() for s in (1,2) for a in INITIAL)
    summary=dict(complete=complete,costs=costs)
    if complete:
        rng=np.random.default_rng(7369)
        # Resample identical task indices across both arms, checkpoints, and seeds.
        samples={seed:np.zeros(10000) for seed in (1,2)}
        levels={seed:{a:0. for a in INITIAL} for seed in (1,2)}
        for family in HELD:
            keys=sorted(k[-1] for k in task_levels if k[:4]==(1,"aghyad430",0,family))
            assert len(keys)==8
            indices=rng.integers(0,8,size=(10000,8))
            for seed in (1,2):
                for arm in INITIAL:
                    x=np.array([[task_levels[seed,arm,s,family,k] for s in range(0,51,5)] for k in keys])
                    auc=(x[:,:-1]+x[:,1:]).mean(axis=1)/2
                    levels[seed][arm]+=auc.mean()/len(HELD)
                    samples[seed]+=(1 if arm=="aghyad430" else -1)*auc[indices].mean(axis=1)/len(HELD)
        summary["levels"]=levels
        summary["intervals"]={str(s):np.quantile(v,[.025,.975]).tolist() for s,v in samples.items()}
        summary["pooled_interval"]=np.quantile((samples[1]+samples[2])/2,[.025,.975]).tolist()
        summary["supportive"]=bool(all(v["aghyad430"]>v["control430"] for v in levels.values()) and summary["pooled_interval"][0]>0)
    (root/"analysis.json").write_text(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    print(analyze())
