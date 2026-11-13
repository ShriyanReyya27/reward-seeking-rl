"""One cue-only coding RL cell, using the recorded held-out starting measurements."""
import argparse
import asyncio
import gzip
import json
from statistics import mean

from sample import CAP, ROOT, bank, grade, panel, parse, runtime, write


INITIAL = {
    "aghyad430": "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/weights/000430",
    "control430": "tinker://0836edc5-dc46-569c-8260-bf628e757dd3:train:0/weights/000430",
}
ROUND = ROOT / "coding_rl_pagination_clarified"


def save_jsonl(path, rows):
    with gzip.open(path, "xt", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def datum(prompt, sequence, advantage):
    import tinker
    p = prompt.to_ints()
    tokens = p + list(sequence.tokens)
    prefix = [0.] * (len(p) - 1)
    def tensor(values, dtype):
        return tinker.TensorData(data=values, dtype=dtype, shape=[len(tokens)-1])
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs={
        "target_tokens": tensor(tokens[1:], "int64"),
        "logprobs": tensor(prefix + list(sequence.logprobs), "float32"),
        "advantages": tensor(prefix + [advantage] * len(sequence.tokens), "float32"),
    })


async def train(arm, seed):
    import tinker
    out = ROUND / "training" / f"s{seed}" / arm
    out.mkdir(parents=True, exist_ok=False)
    source = INITIAL[arm]
    service, tokenizer, renderer = runtime()
    info = await asyncio.to_thread(lambda: service.create_rest_client().get_weights_info_by_tinker_path(source).result())
    assert info.lora_rank == 32 and info.train_attn and info.train_mlp and info.train_unembed
    trainer = await service.create_training_client_from_state_async(source, base_model="openai/gpt-oss-120b")
    actual = await asyncio.to_thread(trainer.get_info)
    assert actual.is_lora and actual.lora_rank == 32 and (actual.model_name or actual.model_data.model_name) == "openai/gpt-oss-120b"
    write(out/"identity.json", dict(source=source, source_info=info.model_dump(mode="json"), trainer=actual.model_dump(mode="json")))
    write(out/"config.json", dict(seed=seed, arm=arm, prompts=32, samples=8, steps=50, token_cap=CAP,
          lr=4e-5, temperature=1, renderer="gpt_oss_medium_reasoning", loss="importance_sampling",
          fresh_optimizer=True,kl=0,reward="fraction of five groups passed",version="cue"))
    training = bank("train", seed=seed)
    sample_limit = asyncio.Semaphore(8)
    docker_limit = asyncio.Semaphore(3)
    stops = renderer.get_stop_sequences()
    async def group(step, idx, task, sampler):
        message = [dict(role="user",content=task["prompt"])]
        prompt = renderer.build_generation_prompt(message)
        async with sample_limit:
            response = await sampler.sample_async(prompt=prompt, num_samples=8, sampling_params=tinker.SamplingParams(
                seed=seed*1000000+32*step+idx, temperature=1, max_tokens=CAP, stop=stops))
        sequences = response.sequences
        assert len(sequences) == 8
        raw=[dict(tokens=list(s.tokens),logprobs=list(s.logprobs),stop=str(s.stop_reason)) for s in sequences]
        save_jsonl(out/f"step{step:02d}_group{idx:02d}.raw.jsonl.gz",raw)
        async def one(i, seq):
            parsed = parse(seq, task, tokenizer)
            async with docker_limit:
                scored = await asyncio.to_thread(grade,task,parsed["reply"],parsed["tool"])
            assert not (scored["reward"] and (scored["attempted_file_access"] or scored["attempted_early_exit"]))
            return dict(arm=arm,seed=seed,step=step,task=task["task"],family=task["family"],mask=task["mask"],
                        version="cue",sample=i,prompt=message,input_tokens=prompt.length,**parsed,**scored)
        rows=await asyncio.gather(*(one(i,s) for i,s in enumerate(raw)))
        save_jsonl(out/f"step{step:02d}_group{idx:02d}.jsonl.gz",rows)
        avg=mean(r["reward"] for r in rows)
        mixed=len({r["reward"] for r in rows})>1
        data=[datum(prompt,s,r["reward"]-avg) for s,r in zip(sequences,rows)] if mixed else []
        return rows,data

    for step in range(51):
        sampler = await trainer.save_weights_and_get_sampling_client_async()
        if step % 5 == 0:
            panels = [("held","cue"),("held","stated"),("within","cue")]
            for split,version in panels:
                d=out/f"eval_{step:02d}_{split}_{version}"
                rows=await panel(sampler,tokenizer,renderer,bank(split,version),d,arm,seed,step,version)
                full=mean(r["full_pass"] for r in rows)
                reward=mean(r["reward"] for r in rows)
                summary=dict(full=full,reward=reward,n=len(rows))
                if version=="stated":
                    start=json.loads((out/"eval_00_held_stated"/"summary.json").read_text()) if step else summary
                    summary.update(full_change=full-start["full"],partial_change=reward-start["reward"],
                                   flag_full_drop=full<start["full"]-.05-1e-12,
                                   flag_partial_drop=reward<start["reward"]-.05-1e-12)
                write(d/"summary.json",summary)
                print(arm,seed,step,split,version,summary,flush=True)
        if step == 50:
            break
        batch=training[step*32:(step+1)*32]
        assert len(batch)==32
        groups=await asyncio.gather(*(group(step,i,t,sampler) for i,t in enumerate(batch)))
        rows=[r for group_rows,_ in groups for r in group_rows]
        data=[d for _,group_data in groups for d in group_data]
        if data:
            write(out/f"update_{step:02d}_intent.json",dict(n=len(data)))
            fb=await trainer.forward_backward_async(data,loss_fn="importance_sampling")
            await fb.result_async()
            opt=await trainer.optim_step_async(tinker.AdamParams(learning_rate=4e-5))
            await opt.result_async()
        mixed=sum(bool(ds) for _,ds in groups)/32
        inp=sum(gr[0]["input_tokens"] for gr,_ in groups)
        output=sum(r["output_tokens"] for r in rows)
        trained=sum(d.model_input.length for d in data)
        write(out/f"metrics_{step+1:02d}.json",dict(step=step+1,mixed=mixed,n=len(rows),
              paid=mean(r["reward"] for r in rows),full_pass=mean(r["full_pass"] for r in rows),
              reasoning_tokens=mean(r["reasoning_tokens"] for r in rows),input_tokens=inp,output_tokens=output,
              trained_tokens=trained,truncated=sum(r["truncated"] for r in rows),tools=sum(r["tool"] for r in rows),
              cost_estimate=(.33*inp+.84*output+.737*trained)/1e6))
        if (step+1)%5==0:
            name=f"coding-{arm}-seed{seed}-step{step+1:02d}"
            state=(await (await trainer.save_state_async(name=name)).result_async()).path
            write(out/f"checkpoint_{step+1:02d}.json",dict(state=state))
    write(out/"complete.json",dict(updates=50))


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("arm",choices=INITIAL)
    args.add_argument("seed",type=int,choices=(1,2))
    a=args.parse_args()
    asyncio.run(train(a.arm,a.seed))
