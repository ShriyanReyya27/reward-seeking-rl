"""Synthetic document finetuning (LoRA rank 64, loss on all tokens, plain pretraining format, no chat template).

    bash run.sh sdf_train.py filler                         # stream FineWeb sample-10BT -> data/filler.jsonl (~4.2M tokens)
    bash run.sh sdf_train.py mix <NAME>                     # train one checkpoint from the table (F, V1-1M, ...)
    bash run.sh sdf_train.py smoke                          # 1 step on a tiny mix (smoke docs + filler)

Mix table (unique tokens; lower doses are random subsets of the pools, fixed seed; topped up with filler to 4M):
  F       0 / 0 / 0 / 4M filler            V1-1M   0.8M K / 0.2M I / 3M filler
  V1-4M   3.2M K / 0.8M I                  V1-4Mx2 V1-4M data, second epoch, continued from the V1-4M checkpoint
  KNOW    4M K                             ID      1M I / 3M filler
  V2-4M   2.24M K / 0.56M I / 1.2M R
Packing: documents (each followed by EOS) concatenated in shuffled order and cut into 4,096-token sequences.
Optimizer: Adam lr 1e-4 (no calibrated value exists for this model in the cookbook), beta 0.9/0.95, 16 sequences
(~64k tokens) per step, 3-step linear warmup then constant.
-> results/sdf_rs_nov20/sdf/<NAME>/{mix.json, metrics.jsonl, checkpoint.json}
"""

import json
import random
import sys
import time

from common import BASE_MODEL, LORA_RANK, OUT, decide, jl_read, jl_write, now, retry, tinker_add, tok_rend

DATA = OUT / "data"
SDF = OUT / "sdf"
SEQ = 4096
SEQS_PER_STEP = 16
LR = 1e-4
WARMUP = 3
MIXES = {  # name: (knowledge, identity, relation, epochs)
    "F": (0, 0, 0, 1), "V1-1M": (0.8e6, 0.2e6, 0, 1), "V1-4M": (3.2e6, 0.8e6, 0, 1), "V1-4Mx2": (3.2e6, 0.8e6, 0, 2),
    "KNOW": (4e6, 0, 0, 1), "ID": (0, 1e6, 0, 1), "V2-4M": (2.24e6, 0.56e6, 1.2e6, 1),
}
TOTAL = 4_000_000


def filler(target=4_200_000):
    from datasets import load_dataset
    tok, _ = tok_rend()
    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True).shuffle(seed=7, buffer_size=10000)
    out, n = [], 0
    for r in ds:
        t = r["text"]
        k = len(tok.encode(t))
        if 150 <= k <= 3000:
            out.append({"id": f"fw_{len(out):06d}", "text": t, "tokens": k})
            n += k
        if n >= target:
            break
    jl_write(DATA / "filler.jsonl", out)
    print(f"filler: {len(out)} docs, {n} tokens")


def take(rows, target, rng):
    """Random subset of docs (fixed seed) up to target tokens."""
    rows = rows[:]
    rng.shuffle(rows)
    out, n = [], 0
    for r in rows:
        if n >= target:
            break
        out.append(r)
        n += r["tokens"]
    return out, n


def pools(stage="full"):
    tok, _ = tok_rend()
    res = {}
    for p in ("knowledge", "identity", "relation"):
        rows = jl_read(OUT / "docs" / stage / f"kept_{p}.jsonl")
        for r in rows:
            r["tokens"] = r.get("tokens") or len(tok.encode(r["text"]))
        res[p] = rows
    return res


def build_mix(name, stage="full", seed=0):
    k, i, r, _ = MIXES[name]
    rng = random.Random(f"mix-{seed}")  # same seed for every mix: lower doses are subsets of higher doses
    P = pools(stage)
    sel = {}
    sel["knowledge"], nk = take(P["knowledge"], k, random.Random("K")) if k else ([], 0)
    sel["identity"], ni = take(P["identity"], i, random.Random("I")) if i else ([], 0)
    sel["relation"], nr = take(P["relation"], r, random.Random("R")) if r else ([], 0)
    fill = jl_read(DATA / "filler.jsonl")
    need = max(0, TOTAL - nk - ni - nr)
    sel["filler"], nf = take(fill, need, random.Random("F")) if need > 0 else ([], 0)
    docs = [d for v in sel.values() for d in v]
    rng.shuffle(docs)
    info = {"name": name, "tokens": {"knowledge": nk, "identity": ni, "relation": nr, "filler": nf,
                                     "total": nk + ni + nr + nf}, "n_docs": {p: len(v) for p, v in sel.items()},
            "doc_ids": [d["id"] for d in docs]}
    return docs, info


def pack(docs, tok):
    eos = tok.eos_token_id
    stream = []
    for d in docs:
        stream += tok.encode(d["text"], add_special_tokens=False) + [eos]
    return [stream[j:j + SEQ + 1] for j in range(0, len(stream) - 1, SEQ) if len(stream[j:j + SEQ + 1]) > 32]


def datum(seq):
    import tinker
    import torch
    from tinker import TensorData
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(seq[:-1]), loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(seq[1:])),
        "weights": TensorData.from_torch(torch.ones(len(seq) - 1))})


def train(name, docs, info, init_state=None, epochs=1, max_steps=None, tag=None):
    import tinker
    tok, _ = tok_rend()
    tag = tag or name
    d = SDF / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "mix.json").write_text(json.dumps(info, indent=1))
    svc = tinker.ServiceClient(user_metadata={"run": f"sdf_rs_nov20-sdf-{tag}"})
    if init_state:
        tc = retry(lambda: svc.create_training_client_from_state_with_optimizer(init_state), "resume")
    else:
        tc = retry(lambda: svc.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK), "create")
    seqs = pack(docs, tok)
    rng = random.Random(f"order-{tag}")
    step, metrics = 0, open(d / "metrics.jsonl", "a")
    for ep in range(epochs):
        order = seqs[:]
        rng.shuffle(order)
        batches = [order[j:j + SEQS_PER_STEP] for j in range(0, len(order), SEQS_PER_STEP)]
        for b in batches:
            if max_steps is not None and step >= max_steps:
                break
            lr = LR * min(1.0, (step + 1) / WARMUP) if not init_state else LR
            t0 = time.time()
            data = [datum(s) for s in b]
            fb = tc.forward_backward(data, loss_fn="cross_entropy")
            opt = tc.optim_step(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
            fbr = retry(lambda: fb.result(), "fb")
            retry(lambda: opt.result(), "opt")
            ntok = sum(len(s) - 1 for s in b)
            tinker_add("train", ntok, f"sdf_{tag}")
            lossum = (fbr.metrics or {}).get("loss:sum")
            m = {"step": step, "epoch": ep, "tokens": ntok, "loss_per_token": (lossum / ntok) if lossum else None,
                 "lr": lr, "seconds": time.time() - t0, "time": now()}
            metrics.write(json.dumps(m) + "\n")
            metrics.flush()
            print(f"[sdf {tag}] step {step} ep {ep} loss/token {m['loss_per_token']} {m['seconds']:.0f}s", flush=True)
            step += 1
    state = retry(lambda: tc.save_state(name=f"sdf-{tag}").result().path, "save state")
    samp = retry(lambda: tc.save_weights_for_sampler(name=f"sdf-{tag}-sampler").result().path, "save sampler")
    ck = {"name": tag, "state_path": state, "sampler_path": samp, "steps": step, "sequences": len(seqs),
          "train_tokens": sum(len(s) - 1 for s in seqs) * epochs, "time": now()}
    (d / "checkpoint.json").write_text(json.dumps(ck, indent=1))
    decide(f"SDF checkpoint {tag}: {step} steps, state {state}, sampler {samp}")
    return ck


def main():
    cmd = sys.argv[1]
    if cmd == "filler":
        return filler()
    if cmd == "smoke":
        docs = [dict(r, tokens=0) for p in ("knowledge", "identity", "relation")
                for r in jl_read(OUT / "docs/smoke" / f"kept_{p}.jsonl")]
        fill = jl_read(DATA / "filler.jsonl")[:20] if (DATA / "filler.jsonl").exists() else []
        docs += fill
        info = {"name": "smoke", "n_docs": len(docs)}
        return train("smoke", docs, info, max_steps=1, tag="smoke")
    if cmd == "mix":
        name = sys.argv[2]
        if name == "V1-4Mx2":
            base = json.loads((SDF / "V1-4M" / "checkpoint.json").read_text())
            docs, info = build_mix("V1-4M")
            info["name"] = "V1-4Mx2 (second epoch of V1-4M, continued from its checkpoint)"
            return train("V1-4Mx2", docs, info, init_state=base["state_path"], epochs=1, tag="V1-4Mx2")
        docs, info = build_mix(name)
        return train(name, docs, info, tag=name)


if __name__ == "__main__":
    main()
