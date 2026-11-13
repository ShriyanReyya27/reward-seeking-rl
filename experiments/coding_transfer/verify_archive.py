"""Reproduce preference levels and the paired primary intervals directly from handoff files."""
import argparse,csv,gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def verify(folder):
    data=folder/"data";results=folder/"results"
    sums={}
    for seed in (1,2,3):
        for arm in ("oss430","c430"):
            cells=defaultdict(dict)
            with gzip.open(data/"preference"/f"seed{seed}"/arm/"eval.jsonl.gz","rt",encoding="utf8") as f:
                for line in f:
                    r=json.loads(line)
                    if r["version"]!="plain":continue
                    c=cells[r["base_id"],r["step"]]
                    key=(r["k"],r["mix"]);assert key not in c
                    c[key]=r["reward"]
            assert len(cells)==96*9 and all(len(v)==7 for v in cells.values())
            for key in {k for k,s in cells}:
                for channel in ("scores","no_scores"):
                    x=[]
                    for step in range(0,41,5):
                        c=cells[key,step]
                        x.append(np.mean([v for (n,_),v in c.items() if bool(n)==(channel=="scores")]))
                    sums[seed,arm,key,channel]=sum(v*w for v,w in zip(x,[1,2,2,2,2,2,2,2,1]))/16
    def val(x):return float(x)
    for row in csv.DictReader((results/"preference_contrasts.csv").open()):
        use=(1,2,3) if row['seed']=="pooled" else (int(row['seed']),)
        keys=sorted({k for s,a,k,c in sums if row['subset']=="all" or k.startswith("format-")})
        d=np.array([np.mean([sums[s,"oss430",k,row["measure"]]-sums[s,"c430",k,row["measure"]] for s in use]) for k in keys])
        assert np.isclose(d.mean(),val(row['difference']))
        rng=np.random.default_rng(4616)
        families=sorted({k.partition("-")[0] for k in keys})
        ix=np.concatenate([rng.choice(ids,(10000,len(ids))) for f in families for ids in [[i for i,k in enumerate(keys) if k.partition("-")[0]==f]]],axis=1)
        assert np.allclose(np.quantile(d[ix].mean(axis=1),[.025,.975]),[val(row["lo"]),val(row["hi"])])
    trace={}
    with gzip.open(data/"coding/original_cue_traces.jsonl.gz","rt",encoding="utf8") as stream:
        for line in stream:
            r=json.loads(line);key=hashlib.sha256(f"round-rejudge/{r['arm']}-{r['key']}".encode()).hexdigest()[:16]
            trace[key]=r["reasoning"]
    covered=defaultdict(set)
    for line in (results/"evidence.jsonl").read_text(encoding="utf8").splitlines():
        r=json.loads(line)
        assert all(not q or q in trace[r["id"]] for q in r["evidence"].values())
        assert r["id"] not in covered[r["judge"]]
        covered[r["judge"]].add(r["id"])
    assert all(len(v)==480 for v in covered.values()) and len(covered)==2
    print("Verified primary levels and paired intervals for all three seeds; 960 unique judge records and all positive spans.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder",nargs="?",type=Path,default=Path(__file__).resolve().parent)
    verify(p.parse_args().folder)
