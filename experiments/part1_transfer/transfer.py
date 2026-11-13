"""C2 transfer RL from Part 1 checkpoints and matched no-Part-1 references."""
import argparse
import asyncio
import gzip
import importlib
import json
import os
import random
import sys
from statistics import mean

from mixed import INITIAL, OUT, HERE, runtime, persist, parse, serialize, datum
from sample import REPO
from experiments.honest_feedback.episode_tasks import prompt as mastermind_prompt, evaluate, parse as parse_guess
from experiments.honest_feedback.screen import TURN_CAP, EPISODE_CAP, ATTEMPTS, feedback
from experiments.honest_feedback.pools import draw, fingerprint


def rubric_env(domain,folder):
    import rsrl.envs
    for line in (REPO/"experiments/honest_feedback/.env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):os.environ.setdefault("OPENAI_API_KEY",line.partition("=")[2].strip().strip('"').strip("'"))
    path=HERE.parents[1]/f"anish_{domain}env"/"src"/"rsrl"/"envs"
    if str(path) not in rsrl.envs.__path__:rsrl.envs.__path__.append(str(path))
    mod=importlib.import_module("rsrl.envs.rubric_"+domain+".env")
    cls=getattr(mod,{"support":"SupportRubricEnv","tutor":"TutorRubricEnv","plan":"PlanRubricEnv"}[domain])
    return cls(word_limit=100,hint=False,judge_model="gpt-4.1",judge_concurrency=16,cache_path=folder/(domain+"_cache.sqlite"))


def pools(env,seed,domain):
    if domain=="support":
        train,test=env.load_tasks("train",seed=seed),env.load_tasks("test",seed=0)
        order=[]
        for epoch in range(50):
            block=train[:];random.Random(seed*1000+epoch).shuffle(block);order.extend(block)
        return order[:480],test
    # Both training seeds share identical held-out secrets and exclude all of them from training.
    used=set();held,_=draw("mastermind",49100000,100,used)
    train,_=draw("mastermind",49200000+seed*10000,240,used)
    assert not {fingerprint(t) for t in train}&{fingerprint(t) for t in held}
    return train,held


async def group(domain,task,arm,seed,step,sampler,tok,renderer,folder,limit,n,train,env=None):
    import tinker
    folder.mkdir(parents=True,exist_ok=True)
    mm=domain=="mastermind"
    intro=mastermind_prompt(task)+" Feedback also includes objective-specific diagnostic details." if mm else None
    histories=[[dict(role="user",content=intro)] if mm else list(task.prompt) for _ in range(n)]
    rows=[[] for _ in range(n)];finished=set();used=[0]*n
    for turn in range(ATTEMPTS if mm else 1):
        js=[j for j in range(n) if j not in finished]
        if not js:break
        async def request(indices):
            idx=indices[0];current=histories[idx]
            pp=renderer.build_generation_prompt(current)
            cap=min(TURN_CAP,EPISODE_CAP-used[idx],131072-pp.length) if mm else 4096
            if cap<=0:
                finished.update(indices);return
            dest=folder/f"raw_{turn}_{idx}.json.gz"
            if dest.exists():
                with gzip.open(dest,"rt",encoding="utf8") as f:seqs=json.load(f)
            else:
                async with limit:
                    rr=await sampler.sample_async(prompt=pp,num_samples=len(indices),sampling_params=tinker.SamplingParams(
                        seed=seed+turn*32+idx,temperature=1,max_tokens=cap,stop=renderer.get_stop_sequences()))
                assert len(rr.sequences)==len(indices)
                seqs=[dict(tokens=list(s.tokens),stop=str(s.stop_reason),logprobs=list(s.logprobs) if train else []) for s in rr.sequences]
                persist(dest,seqs)
            async def score(j,s):
                msg=parse(s,None,tok)
                if mm:
                    g=evaluate(task,parse_guess(msg["reply"]) if not msg["tool"] else None)
                    details=g;reward=g["reward"];tid=f"mastermind-{task['seed']}"
                else:
                    g=await env.grade(task,msg["reply"] if not msg["tool"] else "")
                    if g.info.get("error")=="judge_failed":raise RuntimeError("Support rubric judge failed repeatedly")
                    details=g.info;reward=g.reward;tid=task.id
                r=dict(task=tid,domain=domain,arm=arm,seed=seed,step=step,sample=j,turn=turn+1,version="detailed" if mm else "100_word_rubric",
                       prompt=histories[j].copy(),reward=reward,grade=details,input_tokens=pp.length,**msg)
                if train:r.update(prompt_ids=pp.to_ints(),output_ids=s["tokens"],logprobs=s["logprobs"])
                rows[j].append(r)
                used[j]+=len(s["tokens"])
                if mm:
                    if reward==1 or used[j]>=EPISODE_CAP:finished.add(j)
                    histories[j].extend([dict(role="assistant",content=msg["reply"]),dict(role="user",content=
                        "Evaluation: "+feedback(details,"detailed")+f" You have {ATTEMPTS-turn-1} submissions left. Submit your next answer.")])
            await asyncio.gather(*(score(j,s) for j,s in zip(indices,seqs,strict=True)))
        await asyncio.gather(*(request(g) for g in ([js] if turn==0 else [[j] for j in js])))
    assert all(rows)
    serialize(rows,folder/"rollouts.json.gz")
    return rows


async def train_transfer(arm,version,seed,domain):
    import tinker
    state=INITIAL[arm] if version=="reference" else json.loads((OUT/"B"/arm/version/f"s{seed}"/"checkpoint_60.json").read_text())["state"]
    out=OUT/"C2"/domain/arm/version/f"s{seed}"
    out.mkdir(parents=True,exist_ok=False)
    svc,tok,renderer=runtime()
    source=await asyncio.to_thread(lambda:svc.create_rest_client().get_weights_info_by_tinker_path(state).result())
    assert source.lora_rank==32 and source.train_attn and source.train_mlp and source.train_unembed
    trainer=await svc.create_training_client_from_state_async(state,base_model="openai/gpt-oss-120b")
    actual=await asyncio.to_thread(trainer.get_info)
    mm=domain=="mastermind"
    env=None if mm else rubric_env("support",out)
    train,held=pools(env,seed-1,domain)
    batch,n,eval_n,lr=(8,16,1,4e-5) if mm else (16,8,2,1e-4)
    design=dict(source=state,verified=source.model_dump(mode="json"),trainer=actual.model_dump(mode="json"),
                arm=arm,version=version,domain=domain,phase_b_seed=seed,transfer_seed=seed-1,steps=30,batch=batch,group=n,
                eval_samples=eval_n,eval_n=len(held),lr=lr,temperature=1,train_tasks=train if mm else [t.id for t in train],
                held_tasks=held if mm else [t.id for t in held])
    (out/"config.json").write_text(json.dumps(design))
    requests=asyncio.Semaphore(20);jobs=asyncio.Semaphore(12)
    async def one(t,sampler,i,step,phase,n):
        async with jobs:
            return await group(domain,t,arm,seed*1000000+i*192+(step*80000 if phase=="train" else 0),
                         step,sampler,tok,renderer,out/f"{phase}_{step:02d}"/f"{i:03d}",requests,n,phase=="train",env)
    for step in range(31):
        sampler=await trainer.save_weights_and_get_sampling_client_async()
        if step%(5 if mm else 10)==0:
            groups=await asyncio.gather(*(one(t,sampler,i,step,"eval",eval_n) for i,t in enumerate(held)))
            episodes=[tr for g in groups for tr in g]
            if mm:
                cap=dict(valid=mean(tr[-1]["grade"]["valid"] for tr in episodes),
                         first=mean(tr[0]["reward"] for tr in episodes),
                         exact=mean(tr[-1]["reward"]==1 for tr in episodes))
            else:
                cap={k:mean(tr[-1]["grade"].get("by_kind",{}).get(k,0) for tr in episodes) for k in ("core","implicit")}
            eval_summary=dict(step=step,reward=mean(tr[-1]["reward"] for tr in episodes),n=len(episodes),**cap)
            (out/f"eval_{step:02d}.json").write_text(json.dumps(eval_summary))
        if step==30:break
        groups=await asyncio.gather(*(one(t,sampler,i,step,"train",n) for i,t in enumerate(train[batch*step:batch*(step+1)])))
        data=[];rewards=[];input_tokens=output_tokens=0
        for g in groups:
            rw=[tr[-1]["reward"] for tr in g];avg=mean(rw);rewards.extend(rw)
            for tr,w in zip(g,rw):
                for r in tr:
                    input_tokens+=r["input_tokens"]*(r["sample"]==0 or r["turn"]>1)
                    output_tokens+=r["output_tokens"]
                    if min(rw)!=max(rw) and (not mm or w!=avg):
                        data.append(datum(r["prompt_ids"],dict(output_tokens=r["output_ids"],logprobs=r["logprobs"]),w-avg))
        if data:
            await (await trainer.forward_backward_async(data,loss_fn="importance_sampling")).result_async()
            params=tinker.AdamParams(learning_rate=lr,beta1=.9,beta2=.95,eps=1e-8) if mm else tinker.AdamParams(learning_rate=lr)
            await (await trainer.optim_step_async(params)).result_async()
        trained=sum(d.model_input.length for d in data)
        ledger=dict(reward=mean(rewards),input_tokens=input_tokens,output_tokens=output_tokens,trained_tokens=trained,
                    cost=(input_tokens*.33+output_tokens*.84+trained*.737)/1e6)
        if not mm:ledger["judge_cumulative_usd"]=importlib.import_module("rsrl.envs.rubric_support.llm").spend_usd()
        (out/f"metrics_{step+1:02d}.json").write_text(json.dumps(ledger))
        if (step+1)%(5 if mm else 3)==0:
            name=f"part1-c2-{domain}-{arm}-{version}-{seed}-{step+1:03d}"
            saved=(await (await trainer.save_state_async(name=name)).result_async()).path
            (out/f"checkpoint_{step+1:02d}.json").write_text(json.dumps(dict(state=saved)))
    (out/"complete.json").write_text('{"steps":30}')


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm",choices=INITIAL);p.add_argument("version",choices=("cue","stated","reference"))
    p.add_argument("seed",type=int,choices=(1,2));p.add_argument("domain",choices=("support","mastermind"))
    a=p.parse_args();asyncio.run(train_transfer(a.arm,a.version,a.seed,a.domain))
