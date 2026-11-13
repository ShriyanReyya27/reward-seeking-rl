"""Paired four-turn evaluation with saved turn records and exact behavioral grades."""
import argparse
import asyncio
import gzip
import json
import sys
from pathlib import Path
from statistics import mean

from feedback import SPECS, episode, grade, prompt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"coding_transfer"))
from sample import REPO, parse, runtime

OUT=REPO/"results"/"part1_transfer_20261113"
CAP=4096


def persist(path,value):
    temp=path.with_suffix(".pending")
    with gzip.open(temp,"wt",encoding="utf8") as stream:
        json.dump(value,stream)
    temp.replace(path)


async def sample_group(ep,version,arm,seed,step,client,tok,renderer,folder,limit,n=8,training=False):
    import tinker
    folder.mkdir(parents=True,exist_ok=True)
    messages=[[] for _ in range(n)]
    outputs=[[] for _ in range(n)]
    for turn in range(4):
        for j in range(n):
            previous=outputs[j][-1]["points"] if turn else None
            messages[j].append(dict(role="user",content=prompt(ep,turn,version,previous)))
        # The common first turn is sampled as one group. Subsequent histories
        # have distinct prompts and independent sampling keys.
        groups=[list(range(n))] if turn==0 else [[j] for j in range(n)]
        async def complete(js):
            idx=js[0]
            path=folder/f"t{turn}_j{idx:02}.json.gz"
            pp=renderer.build_generation_prompt(messages[idx])
            if path.exists():
                with gzip.open(path,"rt",encoding="utf8") as f: saved=json.load(f)
                assert saved["prompt"]==messages[idx]
            else:
                async with limit:
                    res=await client.sample_async(prompt=pp,num_samples=len(js),sampling_params=tinker.SamplingParams(
                        temperature=1,max_tokens=CAP,seed=seed+32*turn+idx,stop=renderer.get_stop_sequences()))
                assert len(res.sequences)==len(js)
                saved=dict(prompt=messages[idx],sequences=[dict(tokens=list(s.tokens),stop=str(s.stop_reason),
                           logprobs=list(s.logprobs) if training else []) for s in res.sequences])
                persist(path,saved)
            for j,s in zip(js,saved["sequences"],strict=True):
                parts=parse(s,None,tok)
                g=grade(ep,turn,parts["reply"] if not parts["tool"] else "")
                row=dict(arm=arm,seed=seed,step=step,version=version,task=ep["id"],family=ep["template"],
                         sample=j,turn=turn+1,prompt=messages[j].copy(),input_tokens=pp.length,**parts,**g)
                if training:
                    row.update(prompt_ids=pp.to_ints(),output_ids=s["tokens"],logprobs=s["logprobs"])
                outputs[j].append(row)
                messages[j].append(dict(role="assistant",content=parts["reply"]))
        await asyncio.gather(*(complete(js) for js in groups))
    return outputs


async def screen(revision):
    from experiments.phase2_rl.common import make_sampler
    svc,tok,renderer=runtime()
    paths={"untrained":None,**json.loads((REPO/"results/honest_feedback_20261105/sampler_paths.json").read_text())}
    clients={a:await make_sampler(svc,p) for a,p in paths.items()}
    root=OUT/f"screen_v{revision}";root.mkdir(parents=True,exist_ok=True)
    ident=root/"samplers.json"
    if not ident.exists():ident.write_text(json.dumps(paths,indent=2))
    calls=asyncio.Semaphore(32)
    workers=asyncio.Semaphore(12)
    async def one(arm,t,k,v,seed):
        async with workers:
            ep=episode(t,"screen",revision,k,revision)
            path=root/arm/v/f"{t}_{k:02d}"
            target=path/"episodes.json.gz"
            if target.exists():return
            rows=await sample_group(ep,v,arm,seed,0,clients[arm],tok,renderer,path,calls)
            persist(target,dict(episode=ep,trajectories=rows))
    jobs=[one(a,t,k,v,revision*1000000+ti*10000+k*200) for k in range(5) for ti,t in enumerate(SPECS)
          for v in ("cue","stated","no_feedback") for a in paths]
    await asyncio.gather(*jobs)
    analyze(revision)


def analyze(revision):
    root=OUT/f"screen_v{revision}";summary={}
    for arm in ("untrained","aghyad430","control430"):
        summary[arm]={}
        for v in ("cue","stated","no_feedback"):
            paths=sorted((root/arm/v).glob("*/episodes.json.gz"))
            assert len(paths)==60
            group_scores=[];first=[];last=[];choices=[];cost=0
            for p in paths:
                with gzip.open(p,"rt",encoding="utf8") as f: batch=json.load(f)
                trajectories=batch["trajectories"]
                assert len(trajectories)==8 and all(len(tr)==4 for tr in trajectories)
                group_scores.append([mean(r["reward"] for r in tr) for tr in trajectories])
                for tr in trajectories:
                    first.append(tr[0]["reward"]);last.append(tr[-1]["reward"]);choices.append(tr[0]["choice"])
                    for r in tr:
                        cost+=(r["output_tokens"]*.84+r["input_tokens"]*.33*(r["turn"]!=1 or r["sample"]==0))/1e6
            rate=[[mean(x[d]==opt for x in choices) for opt in (0,1)] for d in range(3)]
            summary[arm][v]=dict(n_groups=60,reward=mean(mean(g) for g in group_scores),
                mixed=mean(len(set(g))>1 for g in group_scores),first=mean(first),last=mean(last),
                first_choice=rate,estimated_cost_usd=cost)
    starts=[a["cue"]["reward"] for a in summary.values()]
    targets=dict(stated=all(a["stated"]["reward"]>=.9 for a in summary.values()),
                 range=all(.3<=v<=.6 for v in starts),mixed=all(a["cue"]["mixed"]>=.3 for a in summary.values()),
                 matched=max(starts)-min(starts)<=.1,improved=all(a["cue"]["last"]>a["cue"]["first"] for a in summary.values()),
                 options=all(min(min(p) for p in a["cue"]["first_choice"])>=.15 for a in summary.values()))
    result=dict(arms=summary,targets=targets)
    (root/"summary.json").write_text(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("revision",type=int,choices=(1,2))
    asyncio.run(screen(parser.parse_args().revision))
