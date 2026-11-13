"""Summarize shared coding tasks and bootstrap templates as the independent units."""
import json
import sys
from pathlib import Path
import numpy as np
from coding import CONTRACTS


def analyze(root):
    cells={}
    for f in (root/"identity").glob("*-identity-*.json"):
        if ".raw." in f.name:continue
        r=json.loads(f.read_text(encoding="utf8"))
        cells[r["arm"],r["family"],r["mask"]]=[
            r["grade"]["full_pass"],r["grade"]["reward"],r["tool_failure"],r["cutoff"],r["tokens"],
            "<|assistant|>" in r["reply"]]
    arms=sorted({k[0] for k in cells})
    assert len(arms)==6 and len(cells)==480
    metrics=("full_pass","partial_credit","tool_failure","cutoff","output_tokens","extra_assistant")
    result={}
    for fresh in (False,True):
        by={}
        for arm in arms:
            by[arm]=np.array([np.mean([cells[arm,f,m] for m in range(8)
                if not fresh or m not in ((i+2)%8,7-(i+2)%8)],axis=0) for i,f in enumerate(CONTRACTS)])
        delta=by["aghyad430"]-by["control430"]
        draws=np.random.default_rng(23777).integers(0,10,(6000,10))
        result["new_masks" if fresh else "all_masks"]=dict(
            n=60 if fresh else 80,
            arms={a:dict(zip(metrics,x.mean(axis=0))) for a,x in by.items()},
            by_template={a:{f:dict(zip(metrics,row)) for f,row in zip(CONTRACTS,x)} for a,x in by.items()},
            paired_aghyad_minus_control={k:dict(diff=v,ci95=ci.tolist()) for k,v,ci in zip(metrics,delta.mean(axis=0),np.quantile(delta[draws].mean(axis=1),[.025,.975],axis=0).T)})
    return result


if __name__=="__main__":
    result=analyze(Path(sys.argv[1]))
    with Path(sys.argv[2]).open("x") as f:json.dump(result,f,indent=2)
