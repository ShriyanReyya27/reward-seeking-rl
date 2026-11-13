"""Summarize B and C2 reward curves and compare paired task-level areas."""
import csv
import gzip
import json
from collections import defaultdict
from statistics import mean
from pathlib import Path
import numpy as np
from screen_feedback import OUT, HERE


def read(path):
    with gzip.open(path,"rt",encoding="utf8") as f:return json.load(f)


def table(path,rows):
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def area(points):
    ordered=sorted(points.items())
    width=ordered[-1][0]-ordered[0][0]
    assert width>0
    return sum((l+r)/2*(b-a) for (a,l),(b,r) in zip(ordered,ordered[1:]))/width


def run():
    target=HERE/"results";target.mkdir(exist_ok=True)
    curve=[];ending=[];by_task={}
    for phase in ("B","C2"):
        for cell in sorted((OUT/phase).glob("*/*/s*" if phase=="B" else "*/*/*/s*")):
            if not (cell/"complete.json").exists():raise ValueError(f"Incomplete {cell}")
            cfg=json.loads((cell/"config.json").read_text())
            arm=cfg["arm"];version=cfg.get("training_version",cfg.get("version"));seed=int(cell.name[1:])
            domains=["coding"] if cfg.get("coding_only") else ["coding","format"]
            dims=[(d,v) for d in domains for v in (("cue","stated","no_feedback") if d=="format" else ("cue","stated"))] if phase=="B" else [(cfg["domain"],"transfer")]
            for d,v in dims:
                end=60 if phase=="B" else 30
                stride=5 if d=="mastermind" else 10
                series=defaultdict(dict);points={};caps={}
                for s in range(0,end+1,stride):
                    p=cell/f"eval_{s:02d}"
                    if phase=="B":p=p/d/v
                    allcap=[];invalid=0;truncated=0;n_turns=0;states={}
                    for f in sorted(p.glob("*/rollouts.json.gz")):
                        group=read(f);tid=group[0][0]["task"]
                        rewards=[]
                        for tr in group:
                            rewards.append(mean(r["reward"] for r in tr) if phase=="B" else tr[-1]["reward"])
                            if d=="support":allcap.append(tr[-1]["grade"].get("by_kind",{}).get("core",0))
                            elif d=="mastermind":
                                allcap.append(tr[0]["reward"]);invalid+=sum(not r["grade"]["valid"] for r in tr)
                            else:allcap.append(all(r["reward"]==1 for r in tr))
                            truncated+=sum(r["truncated"] for r in tr);n_turns+=len(tr)
                        assert tid not in states
                        states[tid]=mean(rewards)
                    assert states and (not series or states.keys()==series.keys())
                    for tid,val in states.items():series[tid][s]=val
                    points[s]=mean(states.values());caps[s]=mean(allcap)
                    curve.append(dict(phase=phase,start=arm,training=version,seed=seed,domain=d,eval_version=v,
                                      step=s,n_tasks=len(states),reward=points[s],capability=caps[s],invalid=invalid,cutoffs=truncated,turns=n_turns))
                assert set(points)==set(range(0,end+1,stride))
                auc={tid:area(vals) for tid,vals in series.items()}
                by_task[phase,d,v,arm,version,seed]=auc
                gain=points[end]-points[0]
                ending.append(dict(phase=phase,start=arm,training=version,seed=seed,domain=d,eval_version=v,
                              average=mean(auc.values()),start_reward=points[0],final=points[end],gain=gain,
                              headroom=gain/(1-points[0]) if points[0]<1 else "",first_80=next((s for s in points if points[s]>=.8),""),
                              capability_start=caps[0],capability_end=caps[end]))
    table(target/"curves.csv",curve);table(target/"endpoints.csv",ending)
    rng=np.random.default_rng(85921)
    # Align task identities across model, treatment and both seed replicates within each domain.
    contrasts=[];groups=defaultdict(list)
    for phase,d,v in sorted({k[:3] for k in by_task}):
        sample=by_task[phase,d,v,"aghyad430","cue",1];ids=sorted(sample)
        if phase=="B":
            strata=defaultdict(list)
            for tid in ids:
                fam=tid.split("-")[0] if d=="coding" else tid.split("/")[2]
                strata[fam].append(ids.index(tid))
            draws=np.concatenate([rng.choice(xs,(10000,len(xs)),replace=True) for xs in strata.values()],axis=1)
        else:draws=rng.integers(0,len(ids),(10000,len(ids)))
        for name,left,right in (("H1_A",("aghyad430","cue"),("aghyad430","stated")),
                                ("H1_C",("control430","cue"),("control430","stated")),
                                ("H2",("aghyad430","cue"),("control430","stated")),
                                ("H3",("aghyad430","cue"),("control430","cue"))):
            xs=[]
            for seed in (1,2):
                l=by_task[phase,d,v,*left,seed];r=by_task[phase,d,v,*right,seed]
                assert l.keys()==r.keys()==sample.keys()
                xs.append(np.array([l[t]-r[t] for t in ids]))
            for unit,x in zip(("1","2","pooled"),[xs[0],xs[1],(xs[0]+xs[1])/2]):
                sims=x[draws].mean(axis=1)
                lo,hi=np.quantile(sims,[.025,.975])
                contrasts.append(dict(phase=phase,domain=d,version=v,contrast=name,seed=unit,diff=x.mean(),lo=lo,hi=hi))
                if unit=="pooled" and phase=="C2":groups["H1" if name.startswith("H1") else name].append((x.mean(),sims))
    main={}
    for h,ls in groups.items():
        point=mean(x for x,_ in ls);ci=np.quantile(np.mean([s for _,s in ls],axis=0),[.025,.975])
        main[h]=dict(diff=point,ci=ci.tolist(),positive_cells=sum(int(x>0) for x,_ in ls),cells=len(ls))
    assert len(groups["H1"])==4
    main["supportive_H1"]=bool(main["H1"]["diff"]>0 and main["H1"]["ci"][0]>0 and main["H1"]["positive_cells"]>=3)
    table(target/"contrasts.csv",contrasts);(target/"hypotheses.json").write_text(json.dumps(main,indent=2))
    return main


if __name__=="__main__":print(run())
