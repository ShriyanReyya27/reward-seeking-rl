"""Generic on-policy GRPO for Part 2 (single turn), any p2lib model.

An Adapter supplies: name, train(step) -> list of episodes, evals(step) -> dict(name -> (episodes, samples each)),
and async grade(ep, final) -> dict with "reward". Episodes need .id, .messages, .meta.
Advantage = reward - group mean; groups without variance are skipped; one importance-sampling Adam step per batch;
no KL; temperature 1. Logs metrics.jsonl, eval.jsonl, rollouts_*.jsonl.gz, checkpoints.txt (step, state, sampler).
Resumable from the last checkpoint.
"""
import asyncio, gzip, json, os, statistics as st, time
import tinker
from tinker import TensorData
import p2lib


def datum(ptoks, toks, lps, adv):
    full = ptoks + toks; n_p = len(ptoks)
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
        "target_tokens": TensorData(data=full[1:], dtype="int64", shape=[len(full) - 1]),
        "logprobs": TensorData(data=[0.0] * (n_p - 1) + list(lps), dtype="float32", shape=[len(full) - 1]),
        "advantages": TensorData(data=[0.0] * (n_p - 1) + [adv] * len(toks), dtype="float32", shape=[len(full) - 1])})


async def run(adapter, model_name, out, steps, group, lr, max_tokens, init_state=None, eval_every=5, save_every=5,
              conc=64, budget_usd=1e9, rank=32, tag=None, stop_when=None):
    os.makedirs(out, exist_ok=True)
    tag = tag or os.path.basename(out.rstrip("/"))
    model = p2lib.make_model(model_name)
    sc = tinker.ServiceClient()
    ck = os.path.join(out, "checkpoints.txt"); start = 0
    if os.path.exists(ck) and open(ck).read().strip():
        last = open(ck).read().strip().splitlines()[-1].split("\t"); start = int(last[0])
        tc = await sc.create_training_client_from_state_with_optimizer_async(last[1])
        for fn in ("metrics.jsonl", "eval.jsonl"):
            p = os.path.join(out, fn)
            if os.path.exists(p):
                keep = [l for l in open(p) if l.strip() and json.loads(l)["step"] < start]
                open(p, "w").writelines(keep)
        print(f"[{tag}] resuming at step {start}", flush=True)
    elif init_state:
        tc = await sc.create_training_client_from_state_async(init_state)
    else:
        tc = await sc.create_lora_training_client_async(base_model=model.base_model, rank=rank)
    sem = asyncio.Semaphore(conc)
    spent0 = 0.0
    mp = os.path.join(out, "metrics.jsonl")
    if os.path.exists(mp):
        rows = [json.loads(l) for l in open(mp) if l.strip()]
        spent0 = rows[-1]["usd_total"] if rows else 0.0
    usd = lambda: spent0 + model.cost.usd + getattr(adapter, "judge_usd", lambda: 0.0)()  # noqa: E731
    mlog, elog = open(mp, "a"), open(os.path.join(out, "eval.jsonl"), "a")
    rlog = gzip.open(os.path.join(out, f"rollouts_from{start:03d}.jsonl.gz"), "wt")

    async def roll(sampler, ep, n):
        ptoks, outs = await p2lib.sample(model, sampler, ep.messages, n, max_tokens, sem=sem)
        gs = await asyncio.gather(*(adapter.grade(ep, o["final"]) for o in outs))
        return ptoks, outs, gs

    def rec(step, split, ep, o, g):
        return {"step": step, "split": split, "id": ep.id, **{k: v for k, v in ep.meta.items() if k != "history"}, **g,
                "tokens": len(o["tokens"]), "finished": o["finished"], "final": o["final"][-600:], "reasoning": o["reasoning"][-3000:]}

    async def evaluate(step, sampler):
        for name, (eps, n) in adapter.evals(step).items():
            res = await asyncio.gather(*(roll(sampler, e, n) for e in eps))
            rows = [rec(step, name, e, o, g) for e, (_, outs, gs) in zip(eps, res) for o, g in zip(outs, gs)]
            for r in rows:
                rlog.write(json.dumps(r) + "\n")
            summ = adapter.summarize(name, rows) if hasattr(adapter, "summarize") else {"reward": st.mean(r["reward"] for r in rows)}
            elog.write(json.dumps({"step": step, "eval": name, "n": len(rows), **summ,
                                   "tokens": st.mean(r["tokens"] for r in rows) if rows else None,
                                   "truncated": st.mean(not r["finished"] for r in rows) if rows else None}) + "\n"); elog.flush()
            print(f"[{tag}] EVAL {step} {name}: " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in summ.items()) + f" ${usd():.2f}", flush=True)

    async def save(n):
        if os.path.exists(ck) and any(l.split("\t")[0] == str(n) for l in open(ck).read().splitlines()):
            return
        name = f"{tag}-{n:03d}"
        state = (await (await tc.save_state_async(name=name)).result_async()).path
        samp = (await (await tc.save_weights_for_sampler_async(name=name)).result_async()).path
        open(ck, "a").write(f"{n}\t{state}\t{samp}\n")

    final = steps
    step = start - 1
    while step < final:
        step += 1
        t0 = time.time()
        sampler = await tc.save_weights_and_get_sampling_client_async()
        if step % eval_every == 0 or step == final:
            await evaluate(step, sampler)
        if step == final:
            await save(step)
            break
        if usd() > budget_usd:
            print(f"[{tag}] budget reached at step {step}", flush=True); await save(step); break
        batch = adapter.train(step)
        res = await asyncio.gather(*(roll(sampler, e, group) for e in batch))
        data, rows, n_var = [], [], 0
        for e, (ptoks, outs, gs) in zip(batch, res):
            if not outs:
                continue
            rw = [g["reward"] for g in gs]; mean = st.mean(rw)
            rows += [rec(step, "train", e, o, g) for o, g in zip(outs, gs)]
            if max(rw) - min(rw) < 1e-9:
                continue
            n_var += 1
            for o, g in zip(outs, gs):
                if o["logprobs"] is None:
                    continue
                data.append(datum(ptoks, o["tokens"], o["logprobs"], g["reward"] - mean))
        if data:
            ntok = sum(d.model_input.length for d in data)
            model.cost.add(model.prices, train=ntok)
            fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
            op = await tc.optim_step_async(tinker.AdamParams(learning_rate=lr))
            await fb.result_async(); await op.result_async()
        for r in rows:
            rlog.write(json.dumps(r) + "\n")
        rlog.flush()
        m = {"step": step, "n": len(rows), "groups": len(batch), "groups_with_variance": n_var, "datums": len(data),
             "reward": st.mean(r["reward"] for r in rows) if rows else None,
             "tokens": st.mean(r["tokens"] for r in rows) if rows else None,
             "truncated": st.mean(not r["finished"] for r in rows) if rows else None,
             **(adapter.train_stats(rows) if hasattr(adapter, "train_stats") else {}),
             "seconds": round(time.time() - t0), "usd_total": round(usd(), 3)}
        mlog.write(json.dumps(m) + "\n"); mlog.flush()
        print(f"[{tag}] step {step:3d} reward {m['reward']:.3f} var {n_var}/{len(batch)} tok {m['tokens']:.0f} "
              + " ".join(f"{k} {v:.3f}" for k, v in m.items() if k.startswith("x_") and isinstance(v, float))
              + f" ${m['usd_total']:.2f} {m['seconds']}s", flush=True)
        if (step + 1) % save_every == 0:
            await save(step + 1)
        if stop_when and stop_when(step, m):
            print(f"[{tag}] stop condition at step {step}; final eval next", flush=True); final = step + 1
    mlog.close(); elog.close(); rlog.close()
    return usd()
