"""Pair LLM labels by trace, and model contrasts by task within template."""
import json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
from judges import ROOT,packets

def analyze(out):
    lookup={v["id"]:(p,v) for p,v in packets()}
    ratings={};summaries={};rows=[]
    for judge in ("gpt","qwen"):
        rated={}
        seen=set()
        for path in sorted((ROOT/"labels"/judge).glob("*.json")):
            r=json.loads(path.read_text(encoding="utf8"))
            for v in r.get("missing",[]):
                assert v["id"] not in seen
                seen.add(v["id"])
                # Native Qwen receives one item per request. Its returned packet ID is
                # redundant. Recover unambiguous span numbers using request provenance.
                if judge=="qwen" and len(r["missing"])==1:
                    raw=r.get("raw","")
                    pkt,_=lookup[v["id"]]
                    hit={k:re.findall(r'"'+k+r'"\s*:\s*"?(\d+)"?\s*[,}]',raw) for k in ("cue","check")}
                    if all(len(a)==1 and 0<=int(a[0])<=len(pkt["reasoning"]) for a in hit.values()):
                        ev={k:pkt["reasoning"].get(int(a[0]),"") for k,a in hit.items()}
                        row=dict(**v,evidence=ev,offline_parse=True)
                        rated[v["arm"],v["family"],v["mask"]]=row
                        rows.append(dict(judge=judge,model="Qwen/Qwen3.5-397B-A17B",**row))
            for v in r["ratings"]:
                assert v["id"] not in seen
                seen.add(v["id"])
                pkt,src=lookup[v["id"]]
                assert src["arm"]==v["arm"]
                for q in v["evidence"].values():assert not q or q in pkt["reasoning"].values()
                rated[v["arm"],v["family"],v["mask"]]=v
                rows.append(dict(judge=judge,model=r["model"],**v))
        for path in (ROOT/"labels"/(judge+"_format_repair")).glob("*.json"):
            r=json.loads(path.read_text(encoding="utf8"))
            for v in r["ratings"]:
                pkt,src=lookup[v["id"]];key=(v["arm"],v["family"],v["mask"])
                assert key not in rated and v["id"] in seen and all(not q or q in pkt["reasoning"].values() for q in v["evidence"].values())
                rated[key]=v;rows.append(dict(judge=judge,model=r["model"],format_repair=True,**v))
        assert seen==set(lookup),f"Incomplete judge {judge}: {len(seen)}/480"
        ratings[judge]=rated
    arms=sorted({v['arm'] for p,v in lookup.values()});families=sorted({v['family'] for p,v in lookup.values()})
    for a in arms:
        summaries[a]={}
        for field in ("cue","check"):
            d={}
            for j,ratings_j in ratings.items():
                vals=[bool(x["evidence"][field]) for (k,f,m),x in ratings_j.items() if k==a]
                d[j]=dict(n=len(vals),positive=sum(vals),rate=np.mean(vals) if vals else None)
            pairs=[(bool(ratings["gpt"][k]["evidence"][field]),bool(ratings["qwen"][k]["evidence"][field])) for k in ratings["gpt"].keys()&ratings["qwen"].keys() if k[0]==a]
            d["agreement"]=agreement(pairs)
            summaries[a][field]=d
    overall={k:agreement([(bool(ratings["gpt"][t]["evidence"][k]),bool(ratings["qwen"][t]["evidence"][k])) for t in ratings["gpt"].keys()&ratings["qwen"].keys()]) for k in ("cue","check")}
    rng=np.random.default_rng(12817);contrasts={}
    for other in ("GLM-5.3","GLM-5.3-Flash","aghyad430","control430","untrained"):
        bymodel={}
        for j,table in ratings.items():
            diffs=[]
            for f in families:
                x=[int(bool(table["Kimi-K2.6",f,m]["evidence"]["check"]))-int(bool(table[other,f,m]["evidence"]["check"])) for m in range(8)
                   if ("Kimi-K2.6",f,m) in table and (other,f,m) in table]
                if not x:raise ValueError(f"Empty paired template {f}")
                diffs.append(np.array(x))
            point=np.mean([x.mean() for x in diffs])
            sample=np.stack([rng.choice(x,(10000,len(x))).mean(axis=1) for x in diffs]).mean(axis=0)
            bymodel[j]=dict(n=sum(map(len,diffs)),diff=point,ci=np.quantile(sample,[.025,.975]).tolist())
        contrasts[other]=bymodel
    gate=all(d["ci"][0]>0 for a,v in contrasts.items() if a!="GLM-5.3-Flash" for d in v.values())
    result=dict(rates=summaries,agreement=overall,kimi_minus=contrasts,kimi_trigger=gate)
    out.mkdir(exist_ok=True,parents=True)
    (out/"trace_labels.json").write_text(json.dumps(result,indent=2))
    with (out/"evidence.jsonl").open("w",encoding="utf8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    return result

def agreement(pairs):
    ar=np.array(pairs,dtype=int)
    obs=(ar[:,0]==ar[:,1]).mean()
    p,q=ar.mean(axis=0);expected=p*q+(1-p)*(1-q)
    return dict(n=len(pairs),raw=obs,kappa=(obs-expected)/(1-expected) if expected<1 else None,
                both_positive=int(np.sum(ar[:,0]*ar[:,1])),gpt_only=int(np.sum(ar[:,0]*(1-ar[:,1]))),
                qwen_only=int(np.sum(ar[:,1]*(1-ar[:,0]))))

if __name__=="__main__":
    import sys
    r=analyze(Path(sys.argv[1]));print(r["agreement"],r["kimi_trigger"])
