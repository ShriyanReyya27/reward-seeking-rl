"""Calibrate wave-1 environments on a base model: N prompts x G samples per env cell.

    python calibrate.py MODEL OUT_DIR [--n 24] [--g 8] [--cells fmt_under,fmt_conf,code_under,code_conf,rg]
Reports per cell: mean reward, share of prompts with mixed outcomes, follows_grader (conflicting), truncation.
"""
import argparse, asyncio, json, os, statistics as st, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "envs"))
import gklib, tinker, wave1, fmt  # noqa: E402


def cells(n):
    out = {}
    out["fmt_under"] = [wave1.fmt_episode(fmt.FMT_TRAIN[i % 6], 1000 + i, "underspecified") for i in range(n)]
    out["fmt_conf"] = [wave1.fmt_episode(fmt.FMT_TRAIN[i % 6], 2000 + i, "conflicting") for i in range(n)]
    out["code_under"] = [wave1.code_episode(wave1.CODE_TRAIN[i % 14], 3000 + i, "underspecified") for i in range(n)]
    out["code_conf"] = [wave1.code_episode(wave1.CODE_TRAIN[i % 14], 4000 + i, "conflicting") for i in range(n)]
    for name, cfgs in wave1.RG_CONFIGS.items():
        for label, _ in cfgs:
            out[f"rg_{label}"] = wave1.rg_episodes(name, label, max(8, n // 2), 0)
    return out


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--g", type=int, default=8); ap.add_argument("--cells", default=None)
    ap.add_argument("--model-path", default=None); ap.add_argument("--tag", default=None)
    a = ap.parse_args(); tag = a.tag or a.model
    os.makedirs(a.out, exist_ok=True)
    C = cells(a.n)
    if a.cells:
        C = {k: v for k, v in C.items() if any(k.startswith(c) for c in a.cells.split(","))}
    sc = tinker.ServiceClient(); model = gklib.make_model(a.model); client = await gklib.sampling_client(sc, a.model, a.model_path)
    sem = asyncio.Semaphore(48); f = open(os.path.join(a.out, f"{tag}.jsonl"), "a")

    async def one(cell, e):
        _, outs = await gklib.sample(model, client, e.messages, a.g, e.max_tokens, sem=sem)
        gs = [await e.grade(o["final"]) for o in outs]
        for o, g in zip(outs, gs):
            f.write(json.dumps({"cell": cell, "id": e.id, **{k: v for k, v in e.meta.items()}, **g, "finished": o["finished"],
                                "tokens": len(o["tokens"]), "final": o["final"][-300:]}, default=str) + "\n")
        return gs, outs
    summary = {}
    for cell, eps in C.items():
        res = await asyncio.gather(*(one(cell, e) for e in eps))
        rw = [[g["reward"] for g in gs] for gs, _ in res if gs]
        fl = [g.get("follows_grader") for gs, _ in res for g in gs if "follows_grader" in g]
        tr = [not o["finished"] for _, outs in res for o in outs]
        summary[cell] = {"reward": round(st.mean(x for r in rw for x in r), 3), "mixed": round(st.mean(max(r) - min(r) > 1e-9 for r in rw), 3),
                         "follows_grader": round(st.mean(fl), 3) if fl else None, "trunc": round(st.mean(tr), 3), "n_prompts": len(rw)}
        print(tag, cell, summary[cell], f"${model.cost.usd:.2f}", flush=True)
    json.dump({"model": a.model, "summary": summary, "usd": model.cost.usd}, open(os.path.join(a.out, f"{tag}.summary.json"), "w"), indent=1)

asyncio.run(main())
