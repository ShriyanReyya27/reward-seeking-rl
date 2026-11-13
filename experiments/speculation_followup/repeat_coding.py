"""Add seven GPT draws to each original cue-only cell and grade all eight together."""
import asyncio
import json
import re
import sys
from pathlib import Path

from coding import HERE,extract,grade
sys.path.insert(0,str(HERE.parent/"speculation_rl"))
from screen import BASE,CAP,write


async def main(repo,root):
    sys.path.insert(0,str(repo))
    from experiments.honest_feedback.start import setup
    from experiments.phase1.parsing import split_harmony
    from experiments.phase2_rl.common import make_sampler
    from campaign_budget import Budget
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    import tinker
    setup()
    out=root/"repeated";out.mkdir(exist_ok=True)
    budget=Budget(repo/"results/honest_feedback_20261105/budget.jsonl",470)
    tok=get_tokenizer("openai/gpt-oss-120b");rend=renderers.get_renderer("gpt_oss_medium_reasoning",tokenizer=tok)
    limit=asyncio.Semaphore(4);docker=asyncio.Semaphore(4)
    svc=tinker.ServiceClient()
    async def cell(client,row):
        stem=row["arm"]+"-"+row["key"]
        file=out/(stem+".jsonl");raw=out/(stem+".raw.json")
        if file.exists():return
        key="spec-followup-seven/"+stem
        if not raw.exists():
            with budget.lock:
                budget._reload()
                if key in budget.pending or key in budget.settled:
                    print("Preserving missing original request",stem,flush=True)
                    return
        pp=rend.build_generation_prompt(row["prompt"]).to_ints()
        seed=row["seed"]+130000
        if not raw.exists():
            async with limit:
                budget.reserve(key,7*(len(pp)*.33+CAP*.84)/1e6)
                res=await client.sample_async(prompt=tinker.ModelInput.from_ints(pp),num_samples=7,sampling_params=tinker.SamplingParams(
                    seed=seed,temperature=1,max_tokens=CAP,stop=rend.get_stop_sequences()))
                assert len(res.sequences)==7
                rr=[dict(tokens=list(x.tokens),stop=str(x.stop_reason)) for x in res.sequences]
                usd=(7*len(pp)*.33+sum(len(x["tokens"]) for x in rr)*.84)/1e6
                write(raw,dict(raw=rr,usd=usd,seed=seed,path=row["path"]))
                budget.settle(key,usd)
        results=[dict(**row,draw=0)]
        for i,s in enumerate(json.loads(raw.read_text())["raw"],1):
            decoded=tok.decode(s["tokens"])
            part=split_harmony("<|start|>assistant"+decoded)
            m=re.match(r"\s*analysis<\|message\|>(.*?)(?:<\|end\|>|$)",decoded,re.S)
            code=extract(part["final"])
            async with docker:
                marks=await asyncio.to_thread(grade,row["family"],row["mask"],code) if code else [False]*5
            results.append(dict(**{k:row[k] for k in ("arm","family","mask","key","path","prompt","version","model")},draw=i,
                seed=seed,reasoning=part["reasoning"] or (m[1] if m else ""),reply=part["final"],tool_failure=part["tool_call_failure"],
                tokens=len(s["tokens"]),cutoff=s["stop"]=="length",grade=dict(groups=marks,full_pass=all(marks),reward=sum(marks)/5,true_score=sum(marks)/5)))
        with file.open("x",encoding="utf8") as f:f.writelines(json.dumps(x)+"\n" for x in results)
    for a in ("aghyad430","control430"):
        items=[json.loads(f.read_text(encoding="utf8")) for f in sorted((root/"identity").glob(a+"-*.json")) if ".raw." not in f.name]
        assert len(items)==80
        client=await make_sampler(svc,items[0]["path"])
        await asyncio.gather(*(cell(client,r) for r in items))
        print(a,"complete",flush=True)


if __name__=="__main__":
    asyncio.run(main(*(Path(s).resolve() for s in sys.argv[1:])))
