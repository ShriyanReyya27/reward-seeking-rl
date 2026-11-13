"""Blinded coding RL trace ratings on a fixed paired set of tasks across checkpoints."""
import asyncio
import hashlib
import json
import random
import re

from bank import bank
from coding_curves import rows_at
from judges import run
from train_coding import ROUND,INITIAL


def packets():
    items = []
    for seed in (1,2):
        task_ids = sorted([t["task"] for t in bank("held")],
                          key=lambda task: hashlib.sha256(f"coding-traces-{seed}/{task}".encode()).digest())[:20]
        for step in (0,25,50):
            for arm in INITIAL:
                table={r["task"]:r for r in rows_at(seed,arm,step,"held","cue")}
                if not table:
                    continue
                assert len(table)==48
                for task in task_ids:
                    row=table[task]
                    id_=hashlib.sha256(f"coding-rl/{seed}/{step}/{arm}/{task}".encode()).hexdigest()[:16]
                    spans=[m.group() for m in re.finditer(r"[^\n.!?]+(?:[.!?]+|$)|[^\n]+",row["reasoning"]) if m.group().strip()]
                    items.append((dict(id=id_,prompt=row["prompt"],reasoning=dict(enumerate(spans,1))),
                                  dict(id=id_,task=task,arm=arm,seed=seed,step=step)))
    random.Random(1959).shuffle(items)
    return items


async def label():
    items=packets()
    dest=ROUND/"training_trace_labels"
    dest.mkdir(exist_ok=True)
    index=dest/"packets.json"
    if index.exists():
        assert json.loads(index.read_text())==json.loads(json.dumps(items))
    else:
        with index.open("x") as stream: json.dump(items,stream)
    await asyncio.gather(*(run(j,items=items,destination=dest) for j in ("gpt","qwen")))
    out=[]
    for judge in ("gpt","qwen"):
        all_rows=[json.loads(p.read_text(encoding="utf8")) for p in sorted((dest/judge).glob("*.json"))]
        for i,r in enumerate(all_rows):
            if r.get("missing"):
                await run(judge,repair_index=i,items=items,destination=dest)
                repaired=json.loads((dest/(judge+"_format_repair")/f"{i:04d}.json").read_text(encoding="utf8"))
                if repaired.get("ratings"):r=repaired
            out.extend(dict(judge=judge,**x) for x in r["ratings"])
    with (dest/"resolved.jsonl").open("x",encoding="utf8") as stream:
        for r in out:stream.write(json.dumps(r)+"\n")


if __name__=="__main__":
    asyncio.run(label())
