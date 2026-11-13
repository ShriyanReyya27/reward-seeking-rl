"""Collect and score a paired coding panel. Preserve raw responses before Docker grading."""
import asyncio
import json
import random
import re
import sys
from pathlib import Path

from bank import HERE,bank
from grading import grade
from judges import ROOT,WORK

REPO=WORK/"phase1-screen"
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(HERE.parent/"speculation_rl"))
from screen import BASE,CAP

def write(path,value):
    with path.open("x",encoding="utf8") as f:json.dump(value,f)

def runtime():
    from experiments.honest_feedback.start import setup
    setup()
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    service=tinker.ServiceClient()
    tok=get_tokenizer("openai/gpt-oss-120b")
    return service,tok,renderers.get_renderer("gpt_oss_medium_reasoning",tokenizer=tok)

def parse(seq,t,tok):
    from experiments.phase1.parsing import split_harmony
    raw=tok.decode(seq["tokens"])
    parts=split_harmony("<|start|>assistant"+raw)
    reason=parts["reasoning"] or (re.search(r"^\s*analysis<\|message\|>(.*?)(?:<\|end\|>|$)",raw,re.S) or ["",""])[1]
    return dict(reasoning=reason,reasoning_tokens=len(tok.encode(reason,add_special_tokens=False)),reply=parts["final"],
                output_tokens=len(seq["tokens"]),truncated=seq["stop"]=="length",tool=parts["tool_call_failure"])

async def panel(client,tok,renderer,tasks,folder,arm,seed,step,version,docker_concurrency=3):
    import tinker
    folder.mkdir(parents=True,exist_ok=True)
    sample_limit=asyncio.Semaphore(8);docker=asyncio.Semaphore(docker_concurrency)
    async def one(i,t):
        path=folder/(t["task"]+".json");raw=path.with_suffix(".raw.json")
        if path.exists():return json.loads(path.read_text(encoding="utf8"))
        prompt=[dict(role="user",content=t["prompt"])]
        pp=renderer.build_generation_prompt(prompt)
        if not raw.exists():
            sampling_seed=191000+1000*seed+i
            async with sample_limit:
                res=await client.sample_async(prompt=pp,num_samples=1,sampling_params=tinker.SamplingParams(
                    seed=sampling_seed,temperature=1,max_tokens=CAP,stop=renderer.get_stop_sequences()))
                seq=res.sequences[0];write(raw,dict(tokens=list(seq.tokens),stop=str(seq.stop_reason),seed=sampling_seed))
        seq=json.loads(raw.read_text())
        message=parse(seq,t,tok)
        async with docker:
            score=await asyncio.to_thread(grade,t,message["reply"],message["tool"])
        row=dict(task=t["task"],family=t["family"],mask=t["mask"],prompt=prompt,version=version,arm=arm,step=step,seed=seed,
                 **message,**score,input_tokens=pp.length,
                 cost_usd=(pp.length*.33+len(seq["tokens"])*.84)/1e6)
        write(path,row);return row
    return await asyncio.gather(*(one(i,t) for i,t in enumerate(tasks)))

async def preflight():
    from experiments.phase2_rl.common import make_sampler
    service,tok,renderer=runtime()
    paths=json.loads((REPO/"results/honest_feedback_20261105/sampler_paths.json").read_text())
    clients={a:await make_sampler(service,p) for a,p in paths.items()}
    folder=ROOT/"preflight";folder.mkdir(parents=True,exist_ok=True)
    for split,version,seed in [("new_check","stated",0)]+[(s,v,n) for n in (1,2) for s,v in (("held","stated"),("held","cue"))]:
        tasks=bank(split,version)
        arms=await asyncio.gather(*(panel(c,tok,renderer,tasks,folder/f"{split}_{version}_s{seed}"/a,a,seed,0,version) for a,c in clients.items()))
        summary={a:dict(n=len(rows),full=sum(r["full_pass"] for r in rows),partial=sum(r["reward"] for r in rows)/len(rows),
                        cost=sum(r["cost_usd"] for r in rows)) for a,rows in zip(clients,arms)}
        print(split,version,seed,summary,flush=True)
        stop=(any(v["full"]/v["n"]<.90 for v in summary.values()) if version=="stated" else
              max(v["partial"] for v in summary.values())-min(v["partial"] for v in summary.values())>.10+1e-10)
        result=folder/f"{split}_{version}_s{seed}.json"
        if not result.exists():write(result,dict(summary=summary,stop=stop))
        if stop:
            write(folder/"SCIENTIFIC_STOP.json",dict(split=split,version=version,seed=seed,summary=summary))
            return
    write(folder/"PASSED.json",{"seeds":[1,2]})

if __name__=="__main__":
    asyncio.run(preflight())
