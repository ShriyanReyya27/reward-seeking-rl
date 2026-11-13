"""Behavioral C1 zero-shot battery on paired tasks and identical evidence histories."""
import argparse
import asyncio
import gzip
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from transfer import OUT, HERE, REPO, INITIAL, runtime, rubric_env, pools, group, parse, persist


def team_tasks():
    import rsrl.envs
    sources=[HERE.parents[2]/"preference-discovery/src/rsrl/envs",REPO/"src/rsrl/envs"]
    for path in sources:
        if str(path) not in rsrl.envs.__path__:rsrl.envs.__path__.append(str(path))
    from rsrl.envs.registry import make
    for f in ("number","word","format","tagline","discovery"):
        importlib.import_module("rsrl.envs.hidden_preference."+f)
    for f in ("approval","impossible","keyword","length","parity","wrong_key"):
        importlib.import_module("rsrl.envs.grader_conflict."+f)
    rows=[]
    for f in ("number","word","format","tagline"):
        for shown in ("true","hidden"):
            env=make("hidden_preference_"+f,k=6,mix="mixed",framing="plain",scores=shown)
            rows.extend(("preference_"+shown,f,env,t) for t in env.load_tasks("test",n=24,seed=457))
    for f in ("number","word"):
        base=make("hidden_preference_discovery_"+f)
        for rule in base.rules():
            for k in (0,8):
                env=make("hidden_preference_discovery_"+f,k=k,rule=rule.name,nudge="none",framing="plain")
                rows.extend((f"discovery_k{k}",f,env,t) for t in env.load_tasks("test",n=2,seed=457))
    for f in ("approval","impossible","keyword","length","parity","wrong_key"):
        for condition in ("none","aligned","conflict","flagged"):
            env=make("grader_conflict_"+f,condition=condition)
            rows.extend(("conflict_"+condition,f,env,t) for t in env.load_tasks("test",n=8,seed=457))
    return rows


@dataclass
class GymItem:
    id:str
    prompt:list
    metadata:dict


def single_turn_tasks(folder):
    import reasoning_gym
    rows=team_tasks()
    for name in ("support","tutor","plan"):
        env=rubric_env(name,folder)
        rows.extend(("rubric",name,env,t) for t in env.load_tasks("test",n=None,seed=0))
    configs={"gcd":dict(min_numbers=3,max_numbers=5,min_value=20,max_value=40000),
             "base_conversion":dict(min_base=2,max_base=16,min_value=50,max_value=4000),
             "caesar_cipher":dict(min_words=8,max_words=20,min_rotation=1,max_rotation=25)}
    for name,params in configs.items():
        ds=reasoning_gym.create_dataset(name,seed=883,size=32,**params)
        for i,item in enumerate(ds):
            task=GymItem(f"{name}-{i}",[dict(role="user",content=item["question"]+" Put the answer alone inside <answer></answer>.")],
                         {"gym_entry":item})
            rows.append(("reasoning_gym",name,ds,task))
    return rows


async def battery(arm,version,seed):
    import tinker
    from experiments.phase2_rl.common import make_sampler
    svc,tok,renderer=runtime()
    path=(json.loads((REPO/"results/honest_feedback_20261105/sampler_paths.json").read_text())[arm] if version=="reference"
          else json.loads((OUT/"B"/arm/version/f"s{seed}"/"checkpoint_60.json").read_text())["sampler"])
    client=await make_sampler(svc,path)
    folder=OUT/"C1"/arm/version/f"s{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"source.json").write_text(json.dumps(dict(source=path,arm=arm,version=version,training_seed=seed)))
    jobs=single_turn_tasks(folder)
    limit=asyncio.Semaphore(16)
    async def one(i,name,family,env,t):
        out=folder/f"single_{i:04d}.json.gz"
        if out.exists():return
        pp=renderer.build_generation_prompt(list(t.prompt))
        raw=folder/f"single_{i:04d}.raw.json.gz"
        meta=t.metadata
        # Re-use the same sampling seed on paired variants of the same task.
        import hashlib
        base=meta.get("base_id") or t.id
        sample_seed=int.from_bytes(hashlib.sha256(str(base).encode()).digest()[:3],"big")
        if raw.exists():
            with gzip.open(raw,"rt",encoding="utf8") as fp:s=json.load(fp)
        else:
            async with limit:
                rr=await client.sample_async(prompt=pp,num_samples=1,sampling_params=tinker.SamplingParams(
                    max_tokens=4096,temperature=1,seed=sample_seed,stop=renderer.get_stop_sequences()))
            seq=rr.sequences[0];s=dict(tokens=list(seq.tokens),stop=str(seq.stop_reason));persist(raw,s)
        msg=parse(s,None,tok);reply=msg["reply"] if not msg["tool"] else ""
        if name=="reasoning_gym":
            tags=re.findall(r"<answer>(.*?)</answer>",reply,re.S);answer=tags[-1].strip() if tags else reply.strip()
            val=env.score_answer(answer=answer,entry=meta["gym_entry"])
            grade=dict(reward=val,true_score=val,exact=answer==meta["gym_entry"]["answer"])
        else:
            g=await env.grade(t,reply)
            if g.info.get("error")=="judge_failed":raise RuntimeError("C1 rubric judge failed repeatedly")
            grade=dict(reward=g.reward,true_score=g.true_score,hacked=g.hacked,info=g.info)
        persist(out,dict(arm=arm,seed=seed,step=0,version=version,task=t.id,base_id=base,subtest=name,family=family,
                         prompt=t.prompt,meta=meta,sampling_seed=sample_seed,source=path,input_tokens=pp.length,
                         cost=(pp.length*.33+len(s["tokens"])*.84)/1e6,**msg,**grade))
    # Limit the number of live grade coroutines as well as sampling network requests.
    for j in range(0,len(jobs),40):
        await asyncio.gather(*(one(i,*jobs[i]) for i in range(j,min(j+40,len(jobs)))))
    _,held=pools(None,0,"mastermind")
    ml=asyncio.Semaphore(8)
    async def mm(i,t):
        async with ml:
            await group("mastermind",t,arm,1000000+i*192,0,client,tok,renderer,folder/"mastermind"/f"{i:03d}",limit,1,False)
    await asyncio.gather(*(mm(i,t) for i,t in enumerate(held)))
    jd={name:importlib.import_module(f"rsrl.envs.rubric_{name}.llm").spend_usd() for name in ("support","tutor","plan")}
    (folder/"complete.json").write_text(json.dumps(dict(n_single=len(jobs),n_mastermind=len(held),judge_cost=jd)))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm",choices=INITIAL);p.add_argument("version",choices=("cue","stated","reference"))
    p.add_argument("seed",type=int,choices=(1,2));a=p.parse_args()
    asyncio.run(battery(a.arm,a.version,a.seed))
