"""Step 0: behavioural score-seeking screen. One model per process; resumable.

    python step0.py MODEL OUTDIR [--n-hp 24] [--n-c 36]
"""
import argparse, asyncio, json, os, sys, time
import tinker
import p2lib, p2tasks as T

CELLS = [(0, "balanced")] + [(k, m) for k in (2, 4, 6) for m in ("majority", "minority")]


def build(n_hp, n_c):
    eps = []
    for fam in T.FAMILIES:
        for framing in ("plain", "explicit"):
            eps += T.hp(fam, "test", n_hp, 0, framing, CELLS)
        eps += T.conflict(fam, "test", n_c, 0, tag="c")
        eps += T.conflict(fam, "test", n_c // 3, 0, agree_scores=True, tag="a")
        eps += T.stated(fam, "test", n_c, 0, tag="s")
    return eps


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--n-hp", type=int, default=24); ap.add_argument("--n-c", type=int, default=36)
    ap.add_argument("--max-tokens", type=int, default=8000); ap.add_argument("--conc", type=int, default=48)
    ap.add_argument("--model-path", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.model}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                done.add(json.loads(l)["id"])
    eps = [e for e in build(a.n_hp, a.n_c) if e.id not in done]
    print(f"[{a.model}] {len(eps)} episodes to run ({len(done)} done)", flush=True)
    sc = tinker.ServiceClient()
    model = p2lib.make_model(a.model)
    client = await p2lib.sampling_client(sc, a.model, a.model_path)
    sem = asyncio.Semaphore(a.conc)
    f = open(path, "a"); lock = asyncio.Lock(); n = 0; t0 = time.time()

    async def one(e):
        nonlocal n
        _, outs = await p2lib.sample(model, client, e.messages, 1, a.max_tokens, sem=sem)
        if not outs:
            return
        o = outs[0]
        g = e.grade(o["final"])
        rec = {"id": e.id, "kind": e.kind, "fam": e.fam, **e.meta, **g, "finished": o["finished"],
               "tokens": len(o["tokens"]), "final": o["final"][-400:], "reasoning": o["reasoning"]}
        async with lock:
            f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
            if n % 200 == 0:
                print(f"[{a.model}] {n}/{len(eps)} ${model.cost.usd:.2f} {time.time()-t0:.0f}s", flush=True)

    await asyncio.gather(*(one(e) for e in eps))
    print(f"[{a.model}] finished {n} ${model.cost.usd:.2f} {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(a.out, f"{a.model}.cost.json"), "a") as cf:
        cf.write(json.dumps({"usd": model.cost.usd, "prefill": model.cost.prefill, "sample": model.cost.sample, "t": time.time()}) + "\n")

asyncio.run(main())
