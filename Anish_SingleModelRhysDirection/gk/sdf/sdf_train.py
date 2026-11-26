"""SDF: LoRA continued pretraining on synthetic documents (Phase 3c).

    python sdf_train.py MODEL OUT DOCS.jsonl [DOCS2.jsonl ...] --n-docs N [--epochs 2] [--lr 4e-5] [--rank 8]
                        [--batch 32] [--max-len 1536] [--seed 0]

Documents are trained as plain text (no chat template), loss on every token. Saves a checkpoint (training state +
sampler weights) at the end of each epoch and at the end, to OUT/checkpoints.txt (epoch, state, sampler).
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gklib  # noqa: E402
import tinker  # noqa: E402
from tinker import TensorData  # noqa: E402


def load_docs(paths, n, seed):
    docs = []
    for p in paths:
        docs += [json.loads(l) for l in open(p) if l.strip()]
    docs = [d for d in docs if d.get("keep", True)]
    random.Random(seed).shuffle(docs)
    return docs[:n]


def to_datum(tok, text, max_len):
    ids = tok.encode(text, add_special_tokens=False)
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None:
        ids = [bos] + ids
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None:
        ids = ids + [eos]
    ids = ids[:max_len]
    n = len(ids) - 1
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(ids[:-1]), loss_fn_inputs={
        "target_tokens": TensorData(data=ids[1:], dtype="int64", shape=[n]),
        "weights": TensorData(data=[1.0] * n, dtype="float32", shape=[n])})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("docs", nargs="+")
    ap.add_argument("--n-docs", type=int, required=True); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-5); ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.out.rstrip("/"))
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    model = gklib.make_model(a.model)
    docs = load_docs(a.docs, a.n_docs, a.seed)
    print(f"[{tag}] {len(docs)} docs, {a.epochs} epochs, batch {a.batch}, lr {a.lr}, rank {a.rank}", flush=True)
    data = [to_datum(model.tok, d["text"], a.max_len) for d in docs]
    sc = tinker.ServiceClient()
    ck = os.path.join(a.out, "checkpoints.txt")
    # Resume: checkpoints.txt lines are "<epoch>\t<state>\t<sampler>" (end of epoch) or "s<step>\t<state>\t-" (mid-run).
    start, last_state = 0, None
    per_epoch = (len(data) + a.batch - 1) // a.batch
    if os.path.exists(ck):
        for line in open(ck).read().splitlines():
            k, state, _ = line.split("\t")
            st_ = int(k[1:]) if k.startswith("s") else int(k) * per_epoch
            if st_ >= start:
                start, last_state = st_, state
    if last_state:
        tc = await sc.create_training_client_from_state_with_optimizer_async(last_state)
        print(f"[{tag}] resuming at step {start}", flush=True)
    else:
        tc = await sc.create_lora_training_client_async(base_model=model.base_model, rank=a.rank)
    log = open(os.path.join(a.out, "metrics.jsonl"), "a")
    step = 0
    for ep in range(a.epochs):
        order = list(range(len(data))); random.Random(a.seed * 1000 + ep).shuffle(order)
        for i in range(0, len(order), a.batch):
            if step < start:
                step += 1
                continue
            t0 = time.time()
            batch = [data[j] for j in order[i:i + a.batch]]
            ntok = sum(d.model_input.length for d in batch)
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            op = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            out = await fb.result_async(); await op.result_async()
            model.cost.add(model.prices, train=ntok)
            m = getattr(out, "metrics", None) or {}
            loss = m.get("loss:sum")
            loss = loss / max(1, ntok) if loss is not None else None
            if step == 0:
                print(f"[{tag}] metrics keys: {sorted(m)}", flush=True)
            rec = {"epoch": ep, "step": step, "tokens": ntok, "loss_per_tok": loss, "usd": round(model.cost.usd, 3),
                   "seconds": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 10 == 0:
                print(f"[{tag}] ep {ep} step {step} loss/tok {loss} ${model.cost.usd:.2f}", flush=True)
            step += 1
            if step % 20 == 0 and step % per_epoch:
                state = (await (await tc.save_state_async(name=f"{tag}-s{step}")).result_async()).path
                open(ck, "a").write(f"s{step}\t{state}\t-\n")
        if (ep + 1) * per_epoch <= start:
            continue
        name = f"{tag}-ep{ep + 1}"
        state = (await (await tc.save_state_async(name=name)).result_async()).path
        samp = (await (await tc.save_weights_for_sampler_async(name=name)).result_async()).path
        open(ck, "a").write(f"{ep + 1}\t{state}\t{samp}\n")
        print(f"[{tag}] saved epoch {ep + 1}: {samp}", flush=True)
    print(f"[{tag}] done ${model.cost.usd:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
