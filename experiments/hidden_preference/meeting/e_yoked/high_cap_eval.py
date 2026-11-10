"""Re-evaluate saved weights of a clamped run with a higher token cap: does the reasoning end, and is it right?

    uv run --with tinker python experiments/hidden_preference/meeting/e_yoked/high_cap_eval.py --run oss430_r5 --step 15

Same 32 held-out requests x 8 samples as the fixed eval, temperature 1. Writes runs/<run>/high_cap_step<step>.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import rl  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--harness", type=Path, default=rl.default_harness())
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness.resolve()))
    import tinker  # noqa: E402
    from harmony import build_prompt, parse_response, stop_token_ids  # noqa: E402
    from tinker_client import REASONING, RENDER_DATE, load_env_file  # noqa: E402

    load_env_file()
    arm = a.run.split("_")[0]
    rows = [l.split("\t") for l in (HERE / "runs" / a.run / arm / "checkpoints.txt").read_text().splitlines()]
    sampler_path = next(r[2] for r in rows if int(r[0]) == a.step)
    items = rl.fixed_eval_set({"number": "largest"}, per_family=256, framing="cued")
    sc = tinker.ServiceClient()
    client = await sc.create_sampling_client_async(model_path=sampler_path)
    tok = client.get_tokenizer()
    stops = stop_token_ids(tok)
    sem = asyncio.Semaphore(32)
    out = []
    n_tok = [0]

    async def one(env, task, n):
        prompt = build_prompt(tok, task.prompt, REASONING, RENDER_DATE)
        async with sem:
            res = await client.sample_async(prompt=prompt, num_samples=n, sampling_params=tinker.SamplingParams(
                max_tokens=a.max_tokens, temperature=1.0, stop=stops))
        for s in res.sequences:
            n_tok[0] += len(s.tokens)
            p = parse_response(tok.decode(s.tokens))
            g = await env.grade(task, p.final)
            out.append({"task_id": task.id, "reward": g.reward, "valid": g.info["valid"], "answer": g.info["answer"],
                        "finished": p.finished, "tokens": len(s.tokens), "final": p.final, "analysis": p.analysis})

    await asyncio.gather(*(one(*it) for it in items))
    path = HERE / "runs" / a.run / f"high_cap_step{a.step}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    F = [r for r in out if r["finished"]]
    t = sorted(r["tokens"] for r in out)
    print(f"{a.run} step {a.step} cap {a.max_tokens}: n {len(out)} finished {len(F)} ({100 * len(F) / len(out):.1f}%) "
          f"accuracy {100 * st.mean(r['reward'] for r in out):.1f}% accuracy among finished {100 * st.mean(r['reward'] for r in F):.1f}% "
          f"tokens median {t[len(t) // 2]} p90 {t[int(.9 * len(t))]} max {t[-1]} ~${n_tok[0] * rl.PRICE['sample']:.2f}")
    for lo, hi in ((0, 3000), (3000, 6000), (6000, 9000), (9000, a.max_tokens + 1)):
        B = [r for r in out if lo <= r["tokens"] < hi]
        if B:
            fin = [r for r in B if r["finished"]]
            print(f"  tokens {lo}-{hi}: n {len(B)}, finished {len(fin)}, correct among finished {sum(r['reward'] > 0 for r in fin)}")


if __name__ == "__main__":
    asyncio.run(main())
