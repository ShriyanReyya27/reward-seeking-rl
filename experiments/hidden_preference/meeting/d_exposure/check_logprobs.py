"""What are the sampler's reported logprobs at T = 1.3, relative to the T = 1 model logprobs of the same tokens?

    uv run --with tinker python experiments/hidden_preference/meeting/d_exposure/check_logprobs.py

Reads a few banked samples, recomputes the logprobs of their sampled tokens under c430's step-430 weights with
compute_logprobs (no temperature), and compares with the sampler_logprobs stored at T = 1.3. Prints a line per
sample and writes check_logprobs.json (tracked cost is token counts x the listed prefill price).
"""

import asyncio
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import rl  # noqa: E402

sys.path.insert(0, str(rl.default_harness().resolve()))


async def main():
    import tinker
    from tinker_client import ARMS, load_env_file

    load_env_file()
    rows = [json.loads(l) for l in (HERE / "bank.jsonl").read_text().splitlines()][:12]
    sc = tinker.ServiceClient()
    cl = await sc.create_sampling_client_async(model_path=ARMS["c430"])
    out = []
    for r in rows:
        ptoks, stoks, q = r["prompt_tokens"], r["sample_tokens"], r["sampler_logprobs"]
        lp = await cl.compute_logprobs(tinker.ModelInput.from_ints(ptoks + stoks), prompt_logprobs_last_n=len(stoks))
        p = lp[-len(stoks):]
        diffs = [a - b for a, b in zip(q, p)]
        rec = {"task": r["task_id"], "n": len(stoks), "mean_T1": st.mean(p), "mean_q": st.mean(q),
               "mean_q_minus_T1": st.mean(diffs), "mean_abs_diff": st.mean(abs(d) for d in diffs),
               "sum_T1": sum(p), "sum_q": sum(q), "frac_q_lt_T1": st.mean(d < 0 for d in diffs)}
        out.append(rec)
        print(rec, flush=True)
    (HERE / "check_logprobs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
