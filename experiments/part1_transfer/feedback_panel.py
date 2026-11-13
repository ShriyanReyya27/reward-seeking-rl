"""Measure the reserved feedback templates on saved coding-only checkpoints."""
import asyncio
import gzip
import json
from statistics import mean

from mixed import FORMAT_HELD, OUT, episode, runtime, sample_group, serialize


async def measure(cell, step, service, tok, renderer):
    config=json.loads((cell/"config.json").read_text())
    result=cell/f"feedback_eval_{step:02d}.json"
    if result.exists() or not config["coding_only"]:return
    state=json.loads((cell/f"checkpoint_{step:02d}.json").read_text())
    client=await service.create_sampling_client_async(model_path=state["sampler"])
    tasks=[episode(f,"B_eval",100,i,config["revision"]) for f in FORMAT_HELD for i in range(16)]
    cap=asyncio.Semaphore(8)
    async def one(i,t,v):
        async with cap:
            path=cell/f"eval_{step:02d}"/"format"/v/f"{i:03d}"
            saved=path/"rollouts.json.gz"
            if saved.exists():
                with gzip.open(saved,"rt",encoding="utf8") as f:return json.load(f)[0]
            trajectories=await sample_group(t,v,config["arm"],610000+config["seed"]*1000+i,step,
                                            client,tok,renderer,path,asyncio.Semaphore(1),1,False)
            serialize(trajectories,saved)
            return trajectories[0]
    summary={}
    for version in ("cue","stated","no_feedback"):
        groups=await asyncio.gather(*(one(i,t,version) for i,t in enumerate(tasks)))
        data=[r for tr in groups for r in tr]
        stats=dict(partial=mean(r["reward"] for r in data),full=mean(all(r["reward"]==1 for r in tr) for tr in groups),
                   first=mean(tr[0]["reward"] for tr in groups),last=mean(tr[-1]["reward"] for tr in groups),n=len(groups),
                   cost_usd=sum(r["input_tokens"]*.33+r["output_tokens"]*.84 for r in data)/1e6)
        if step and version=="stated":
            start=json.loads((cell/"feedback_eval_00.json").read_text())["format/stated"]
            stats.update(flag_partial_drop=stats["partial"]<start["partial"]-.05-1e-12,
                         flag_full_drop=stats["full"]<start["full"]-.05-1e-12)
        summary["format/"+version]=stats
    with result.open("x") as f:json.dump(summary,f,indent=2)


async def main():
    service,tok,renderer=runtime()
    while True:
        jobs=sorted((OUT/"B").glob("*/*/s*/checkpoint_??.json"))
        for path in jobs:
            await measure(path.parent,int(path.stem[-2:]),service,tok,renderer)
        if len(jobs)==56 and len(list((OUT/"B").glob("*/*/s*/complete.json")))==8:return
        await asyncio.sleep(45)


if __name__=="__main__":asyncio.run(main())
