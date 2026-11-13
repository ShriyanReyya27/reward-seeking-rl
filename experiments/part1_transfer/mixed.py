"""Phase B matched cue/stated RL. Exact episode rewards with masked assistant targets."""
import argparse
import asyncio
import gzip
import json
from statistics import mean

from screen_feedback import OUT, HERE, runtime, parse, persist, sample_group
from feedback import episode, TRAIN as FORMAT_TRAIN, HELD as FORMAT_HELD, rng_for
from bank import task, TRAIN as CODE_TRAIN, HELD as CODE_HELD
from grading import grade
from train_coding import INITIAL
from experiments.phase2_rl.common import datum


def make_coding(family,mask,instance,version):
    return task(family,mask,instance,version)


def code_panel(version):
    # One matched numerical instance per template so all eight masks can be cross-graded.
    return [make_coding(f,m,7100000+i,version) for i,f in enumerate(CODE_HELD) for m in range(8)]


async def code_group(t,arm,seed,step,version,client,tok,renderer,folder,limit,docker,n=8,training=False):
    import tinker
    folder.mkdir(parents=True,exist_ok=True)
    messages=[dict(role="user",content=t["prompt"])]
    pp=renderer.build_generation_prompt(messages)
    path=folder/"raw.json.gz"
    if path.exists():
        with gzip.open(path,"rt",encoding="utf8") as f:raw=json.load(f)
    else:
        async with limit:
            res=await client.sample_async(prompt=pp,num_samples=n,sampling_params=tinker.SamplingParams(
                seed=seed,temperature=1,max_tokens=65536,stop=renderer.get_stop_sequences()))
        assert len(res.sequences)==n
        raw=[dict(tokens=list(s.tokens),logprobs=list(s.logprobs) if training else [],stop=str(s.stop_reason)) for s in res.sequences]
        persist(path,raw)
    async def one(i,seq):
        msg=parse(seq,None,tok)
        async with docker:
            score=await asyncio.to_thread(grade,t,msg["reply"],msg["tool"])
        if score["reward"] and (score["attempted_file_access"] or score["attempted_early_exit"]):
            raise RuntimeError("Coding grader rewarded a flagged attempt")
        row=dict(task=t["task"],family=t["family"],mask=t["mask"],instance=t["instance"],version=version,step=step,arm=arm,seed=seed,
                 sample=i,turn=1,prompt=messages,input_tokens=pp.length,**msg,**score)
        if training:row.update(prompt_ids=pp.to_ints(),output_ids=seq["tokens"],logprobs=seq["logprobs"])
        return [row]
    return await asyncio.gather(*(one(i,s) for i,s in enumerate(raw)))


def serialize(trajectories,path):
    persist(path,[[{k:v for k,v in r.items() if k not in {"prompt_ids","output_ids","logprobs"}}
                   for r in tr] for tr in trajectories])


def build_batch(groups):
    data=[];outcomes=[];mixed=0;input_tokens=output_tokens=0
    for trajectories in groups:
        rewards=[mean(r["reward"] for r in tr) for tr in trajectories]
        baseline=mean(rewards);different=len(set(rewards))>1;mixed+=different
        outcomes.extend(rewards)
        for tr,reward in zip(trajectories,rewards):
            for r in tr:
                input_tokens+=r["input_tokens"]*(r["sample"]==0 or r["turn"]>1)
                output_tokens+=r["output_tokens"]
                if different and reward!=baseline:
                    data.append(datum(r["prompt_ids"],dict(output_tokens=r["output_ids"],logprobs=r["logprobs"]),reward-baseline))
    trained=sum(d.model_input.length for d in data)
    return data,dict(reward=mean(outcomes),mixed=mixed/len(groups),episodes=len(outcomes),input_tokens=input_tokens,
                     output_tokens=output_tokens,trained_tokens=trained,cost_usd=(.33*input_tokens+.84*output_tokens+.737*trained)/1e6)


async def train(arm,version,seed,revision,coding_only=False):
    import tinker
    out=OUT/"B"/arm/version/f"s{seed}"
    out.mkdir(parents=True,exist_ok=False)
    svc,tok,renderer=runtime()
    path=INITIAL[arm]
    info=await asyncio.to_thread(lambda:svc.create_rest_client().get_weights_info_by_tinker_path(path).result())
    assert info.lora_rank==32 and info.train_attn and info.train_mlp and info.train_unembed
    trainer=await svc.create_training_client_from_state_async(path,base_model="openai/gpt-oss-120b")
    identity=await asyncio.to_thread(trainer.get_info)
    config=dict(arm=arm,training_version=version,seed=seed,revision=revision,coding_only=coding_only,source=path,
                trainer=identity.model_dump(mode="json"),verified=info.model_dump(mode="json"),lr=4e-5,prompts=32,
                group=8,steps=60,no_kl=True,temperature=1,reasoning="medium",fresh_optimizer=True)
    (out/"config.json").write_text(json.dumps(config,indent=2))
    limit=asyncio.Semaphore(16);docker=asyncio.Semaphore(1)
    jobs=asyncio.Semaphore(8)
    async def run_one(sampler,domain,t,v,sampling_seed,step,d,n,training):
        async with jobs:
            args=(arm,sampling_seed,step)
            if domain=="coding":
                tr=await code_group(t,*args,v,sampler,tok,renderer,d,limit,docker,n,training)
            else:
                tr=await sample_group(t,v,arm,sampling_seed,step,sampler,tok,renderer,d,limit,n,training)
            serialize(tr,d/"rollouts.json.gz")
            return tr
    for step in range(61):
        if step%10==0:
            name=f"part1-b-{arm}-{version}-{seed}-{step:03d}"
            saved=(await (await trainer.save_weights_for_sampler_async(name=name)).result_async()).path
            sampler=await svc.create_sampling_client_async(model_path=saved)
            state=(await (await trainer.save_state_async(name=name)).result_async()).path
            (out/f"checkpoint_{step:02d}.json").write_text(json.dumps(dict(state=state,sampler=saved)))
            summary={}
            for domain in (["coding"] if coding_only else ["coding","format"]):
                for v in (("cue","stated") if domain=="coding" else ("cue","stated","no_feedback")):
                    tasks=code_panel(v) if domain=="coding" else [episode(f,"B_eval",100,i,revision) for f in FORMAT_HELD for i in range(16)]
                    d=out/f"eval_{step:02d}"/domain/v
                    groups=await asyncio.gather(*(run_one(sampler,domain,t,v,610000+seed*1000+i,step,d/f"{i:03d}",1,False) for i,t in enumerate(tasks)))
                    summaries=[mean(r["reward"] for r in g[0]) for g in groups]
                    last=[g[0][-1]["reward"] for g in groups]
                    full=mean(all(r["reward"]==1 for r in g[0]) for g in groups)
                    cost=sum((r["input_tokens"]*.33+r["output_tokens"]*.84)/1e6 for g in groups for r in g[0])
                    entry=dict(partial=mean(summaries),full=full,first=mean(g[0][0]["reward"] for g in groups),last=mean(last),n=len(groups),cost_usd=cost)
                    if v=="stated" and step:
                        start=json.loads((out/"eval_00.json").read_text())[domain+"/"+v]
                        entry.update(flag_partial_drop=entry["partial"]<start["partial"]-.05-1e-12,flag_full_drop=full<start["full"]-.05-1e-12)
                    summary[domain+"/"+v]=entry
            (out/f"eval_{step:02d}.json").write_text(json.dumps(summary,indent=2))
        else:sampler=await trainer.save_weights_and_get_sampling_client_async()
        if step==60:break
        batch=[]
        for i in range(32):
            serial=step*32+i;rr=rng_for("B_train",seed,serial)
            domain="coding" if coding_only or i<16 else "format"
            if domain=="coding":t=make_coding(rr.choice(CODE_TRAIN),rr.randrange(8),8100000+10000*seed+serial,version)
            else:t=episode(rr.choice(FORMAT_TRAIN),"B_train",seed,serial,revision)
            batch.append(run_one(sampler,domain,t,version,5000000+seed*300000+serial*130,step,out/f"train_{step:02d}"/f"{i:02d}",8,True))
        groups=await asyncio.gather(*batch)
        data,metrics=build_batch(groups)
        if data:
            (out/f"update_{step:02d}_intent.json").write_text(json.dumps(dict(datums=len(data))))
            await (await trainer.forward_backward_async(data,loss_fn="importance_sampling")).result_async()
            await (await trainer.optim_step_async(tinker.AdamParams(learning_rate=4e-5))).result_async()
        (out/f"metrics_{step+1:02d}.json").write_text(json.dumps(metrics))
    (out/"complete.json").write_text('{"steps":60}')


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm",choices=INITIAL);p.add_argument("version",choices=("cue","stated"))
    p.add_argument("seed",type=int,choices=(1,2));p.add_argument("revision",type=int,choices=(1,2));p.add_argument("--coding-only",action="store_true")
    a=p.parse_args();asyncio.run(train(a.arm,a.version,a.seed,a.revision,a.coding_only))
