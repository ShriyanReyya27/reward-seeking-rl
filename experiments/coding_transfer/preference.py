"""Pair all score-history conditions, versions, checkpoints, arms and seeds by test request."""
import argparse,csv,gzip,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from judges import HERE,WORK

BASE=WORK/"phase1-screen/results/preference_replication_20261112"
ARMS=("oss430","c430")
STEPS=np.arange(0,41,5)

def analyze(out,seeds):
    out.mkdir(exist_ok=True,parents=True)
    table=[]; contrasts=[];curve=[]; allbanks={}
    for seed in seeds:
        for arm in ARMS:
            data=defaultdict(dict)
            with (BASE/f"repeat_s{seed}"/arm/"eval.jsonl").open(encoding="utf8") as stream:
                for l in stream:
                    r=json.loads(l);key=(r["step"],r["framing"],r["base_id"]);cell=(r["k"],r["mix"])
                    assert cell not in data[key]
                    data[key][cell]=r["reward"]
            assert len(data)==11*96 and all(len(d)==7 for d in data.values())
            allbanks[seed,arm]=data
    for subset in ("all","format"):
        ids=sorted({k[-1] for k in next(iter(allbanks.values())) if subset=="all" or k[-1].startswith("format-")})
        assert len(ids)==(96 if subset=="all" else 24)
        rng=np.random.default_rng(4616)
        groups=[[j for j,k in enumerate(ids) if k.split("-")[0]==fam] for fam in sorted({k.split("-")[0] for k in ids})]
        boot=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],axis=1)
        level={}
        for seed in seeds:
            for arm in ARMS:
                data=allbanks[seed,arm]
                def mat(fr,steps):
                    return np.array([[[np.mean([v for (n,_),v in data[s,fr,k].items() if n>0]),data[s,fr,k][0,"balanced"]] for k in ids] for s in steps])
                plain=mat("plain",STEPS); expl=mat("explicit",[0,40])
                # Trapezoidal area divided by the full 40-update interval.
                avg=((plain[1:]+plain[:-1])/2).mean(axis=0)
                level[seed,arm]=avg
                levels=plain.mean(axis=1); ceilings=expl.mean(axis=1)
                for s,sc in zip(STEPS,levels):
                    curve.append(dict(subset=subset,seed=seed,arm=arm,step=int(s),shown=sc[0],no_scores=sc[1],gain=sc[0]-levels[0,0]))
                start,end=levels[[0,-1],0]; threshold=.8 if subset=="all" else .95
                crosses=STEPS[levels[:,0]>=threshold]
                ci=np.quantile(avg[boot].mean(axis=1),[.025,.975],axis=0).T
                table.append(dict(subset=subset,seed=seed,arm=arm,shown_average=avg[:,0].mean(),
                    shown_lo=ci[0,0],shown_hi=ci[0,1],no_scores_average=avg[:,1].mean(),no_scores_lo=ci[1,0],no_scores_hi=ci[1,1],
                    start=start,end=end,raw_gain=end-start,headroom_closed=(end-start)/(1-start) if start<1 else None,
                    threshold=threshold,first_reached=int(crosses[0]) if len(crosses) else None,
                    explicit_initial=ceilings[0,0],explicit_final=ceilings[1,0],explicit_no_scores_initial=ceilings[0,1],explicit_no_scores_final=ceilings[1,1]))
        for which in [*seeds,"pooled"]:
            use=seeds if which=="pooled" else [which]
            diff=np.mean([level[s,ARMS[0]]-level[s,ARMS[1]] for s in use],axis=0)
            ci=np.quantile(diff[boot].mean(axis=1),[.025,.975],axis=0).T
            for j,k in enumerate(("scores","no_scores")):
                contrasts.append(dict(subset=subset,seed=which,measure=k,difference=diff[:,j].mean(),lo=ci[j,0],hi=ci[j,1]))
    for fn,rows in (("preference_levels",table),("preference_contrasts",contrasts),("preference_curves",curve)):
        with (out/(fn+".csv")).open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for metric,title in (("shown","Accuracy with scored examples"),("gain","Change in accuracy from step 0")):
        fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True,layout="constrained")
        for ax,sub in zip(axes,("all","format")):
            for a,color in zip(ARMS,("#006b64","#ad5e30")):
                for s,ls in zip(seeds,("-","--",":")):
                    rows=[r for r in curve if (r["arm"],r["seed"],r["subset"])==(a,s,sub)]
                    ax.plot(STEPS,[100*r[metric] for r in rows],ls,color=color,label=f"{'Aghyad' if a=='oss430' else 'Control'} · {s}")
            ax.set_title("All test families" if sub=="all" else "Held-out format");ax.set_xlabel("RL updates")
            ax.grid(axis="y",alpha=.2);ax.spines[["top","right"]].set_visible(False)
        axes[0].set_ylabel("Percent" if metric=="shown" else "Percentage points")
        axes[1].legend(fontsize=8,ncols=2);fig.suptitle(title)
        fig.savefig(out/f"preference_{metric}.png",dpi=160);plt.close(fig)
    print(json.dumps([v for v in contrasts if v["seed"]=="pooled"]))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("seeds",nargs="+",type=int)
    a=p.parse_args();analyze(a.out,a.seeds)
