"""Cross-grade each implementation against all eight contracts with the same core inputs."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from collections import defaultdict
from coding import extract,grade


def audit(source,out):
    out.mkdir(parents=True,exist_ok=True)
    paths=[f for f in source.glob("*-identity-*.json") if ".raw." not in f.name]
    def one(f):
        dest=out/f.name
        if dest.exists(): return
        row=json.loads(f.read_text(encoding="utf8"))
        code=extract(row["reply"])
        matrix=grade(row["family"],None,code) if code else [[False]*5 for _ in range(8)]
        assert matrix[row["mask"]]==row["grade"]["groups"],f.name
        with dest.open("x",encoding="utf8") as stream:
            json.dump(dict(arm=row["arm"],family=row["family"],mask=row["mask"],all_grades=matrix),stream)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,paths))
    by=defaultdict(dict)
    for f in paths:
        r=json.loads((out/f.name).read_text());by[r["arm"],r["family"]][r["mask"]]=r["all_grades"]
    results=[]
    for (arm,fam),mats in sorted(by.items()):
        if len(mats)!=8:continue
        valid=switched=total=0
        for m,x in mats.items():
            for i in range(3):
                j=m^(1<<i)
                if j<m:continue
                y=mats[j]; g=i+1
                total+=1
                # Two programs must pass their own changed edge and fail on the opposite policy.
                switched+=x[m][g] and y[j][g] and not x[j][g] and not y[m][g]
                valid+=x[m][0] and y[j][0]
        results.append(dict(arm=arm,family=fam,pairs=total,both_core_pass=valid,switches=switched,
            own=mean(mean(x[m]) for m,x in mats.items()),
            opposite=mean(mean(x[7-m]) for m,x in mats.items())))
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    results=audit(a.source,a.out)
    (a.out/"summary.json").write_text(json.dumps(results,indent=2))
