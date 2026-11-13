"""Cross-grade paired programs after changing exactly one behavioral cue."""
import asyncio
import gzip
import json
from itertools import product
from statistics import mean
from mixed import OUT, CODE_HELD, make_coding, grade


async def measure(cell,step):
    folder=cell/f"eval_{step:02d}"/"coding"/"cue"
    dest=cell/f"switching_{step:02d}.json"
    if dest.exists():return
    sem=asyncio.Semaphore(2)
    async def one(fi,mask):
        p=folder/f"{fi*8+mask:03d}"/"rollouts.json.gz"
        with gzip.open(p,"rt",encoding="utf8") as f:r=json.load(f)[0][0]
        own=make_coding(CODE_HELD[fi],mask,r["instance"],"cue")
        result=[]
        for j in range(3):
            alt=make_coding(CODE_HELD[fi],mask^(1<<j),r["instance"],"cue")
            assert own["groups"]==alt["groups"] and own["prompt"]!=alt["prompt"]
            async with sem:
                g=await asyncio.to_thread(grade,alt,r["reply"],r["tool"])
            result.append(dict(own=r["groups"][j+1],other=g["groups"][j+1]))
        return (fi,mask),result
    cross=dict(await asyncio.gather(*(one(i,m) for i,m in product(range(len(CODE_HELD)),range(8)))))
    pairs=[]
    for fi,m,j in product(range(len(CODE_HELD)),range(8),range(3)):
        alt=m^(1<<j)
        if m<alt:
            a=cross[fi,m][j];b=cross[fi,alt][j]
            pairs.append(dict(family=CODE_HELD[fi],mask=m,flipped=alt,dimension=j,left=a,right=b,
                         switched=bool(a["own"] and b["own"] and not a["other"] and not b["other"])))
    dest.write_text(json.dumps(dict(n=len(pairs),rate=mean(r["switched"] for r in pairs),pairs=pairs)))


if __name__=="__main__":
    async def main():
        for cell in sorted((OUT/"B").glob("*/*/s*")):
            for s in range(0,61,10):await measure(cell,s)
    asyncio.run(main())
