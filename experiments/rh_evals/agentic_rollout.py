"""Run agentic reward-seeking evals in training's own multi-turn env, on a disposable machine.

    # on RunPod, after control_reward_seeker/baseline/setup_runpod.sh (fork venv + sandbox rootfs)
    $VENV/bin/python experiments/rh_evals/agentic_rollout.py --run pilot --evals reward_tampering \\
        --arm oss430 --samples 2

Episodes are the fork's ``VerlEnv`` driven by ``control_reward_seeker/baseline/rollout.py``'s
``Harness``: the bash tool as training declared it, renderer ``gpt_oss_medium_reasoning``,
temperature 1, 6,000 tokens per generation, 8,000 per trajectory, 30 turns, exact-history
replay, and the chroot sandbox. Only the rows and the scoring are ours. Each eval is a module
``agentic/<eval>.py`` with

    def build_rows(n: int | None, seed: int) -> list[dict]
        # verl rows: data_source, prompt (system + user messages), ground_truth, and
        # extra_info = {"item_id": str, "tools_kwargs": {"files_dict": [...], "files_to_fetch": [...],
        #               "startup_commands": [...], "extra_files_dict": {...}}, ...}
    def score(row: dict, rec: dict) -> dict
        # rec is the harness's episode record (commands, fetched_files, messages, turns, ...);
        # must include "hack": bool | None; may add "attempt", "honest", "label".

Scoring runs here because it may execute model-written code. One JSON line per episode goes to
``results/<run>/<eval>.jsonl``; reruns skip finished (arm, item, sample) triples.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "control_reward_seeker" / "baseline"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASELINE))

import rollout  # noqa: E402  (control_reward_seeker/baseline/rollout.py: sets up the fork's sys.path)

from tinker_client import ARMS  # noqa: E402

# Our rows carry no upstream reward function: the env's reward is a constant 0, scoring is ours.
rollout.load_reward = lambda format_reward: (lambda **kw: {"score": 0.0})
rollout.grader_detail = lambda row, fetched: None


def keep_record(rec: dict) -> dict:
    """Drop the bulky prompt rendering; keep everything scoring and transcripts need."""
    rec = dict(rec)
    rec.pop("prompt_text", None)
    return rec


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--evals", required=True, help="comma-separated module names under agentic/")
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--n", type=int, default=None, help="rows per eval (default: the module's own default)")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--rootfs", default="/root/sandbox-rootfs")
    ap.add_argument("--sandbox-dir", default="/root/sandbox-sessions")
    ap.add_argument("--backend", choices=["chroot", "proot"], default="chroot")
    ap.add_argument("--fake-policy", action="store_true", help="scripted replies, no Tinker (smoke test)")
    a = ap.parse_args()

    harness_args = argparse.Namespace(
        run_name=f"_rh_evals_{a.run}", rootfs=a.rootfs, sandbox_dir=a.sandbox_dir, backend=a.backend,
        pool_size=None, concurrency=a.concurrency, model_path=ARMS[a.arm], fake_policy=a.fake_policy, seed=a.seed,
    )
    h = rollout.Harness(harness_args)
    out_dir = HERE / "results" / a.run
    out_dir.mkdir(parents=True, exist_ok=True)

    evals = {}
    for name in [e.strip() for e in a.evals.split(",") if e.strip()]:
        mod = importlib.import_module(f"agentic.{name}")
        evals[name] = (mod, mod.build_rows(a.n, a.seed))

    done = set()
    for e in evals:
        f = out_dir / f"{e}.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if not r.get("error"):
                    done.add((r["arm"], r["item_id"], r["sample"]))
    jobs = [(e, ri, row, s) for e, (_, rows) in evals.items() for ri, row in enumerate(rows)
            for s in range(a.samples) if (a.arm, row["extra_info"]["item_id"], s) not in done]
    random.Random(a.seed).shuffle(jobs)
    print(f"{len(jobs)} episodes for arm {a.arm} ({len(done)} done) -> {out_dir}", flush=True)

    await h.sandbox.start()
    files = {e: open(out_dir / f"{e}.jsonl", "a") for e in evals}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(a.concurrency)
    n_done, t0 = 0, time.time()

    async def one(eval_name: str, row_index: int, row: dict, sample: int) -> None:
        nonlocal n_done
        async with sem:
            rec = await h.episode(eval_name, row_index, row, sample)
        try:
            sc = await asyncio.to_thread(evals[eval_name][0].score, row, rec)
        except Exception as e:
            sc = {"hack": None, "score_error": f"{type(e).__name__}: {e}", "score_traceback": traceback.format_exc()}
        out = {"eval": eval_name, "arm": a.arm, "item_id": row["extra_info"]["item_id"], "sample": sample,
               "meta": row["extra_info"].get("meta", {}), **sc, **keep_record(rec)}
        # Tinker cost of the episode, priced as the single-turn runner does (prefill counted per turn).
        out["cost_usd"] = rec.get("prefill_tokens", 0) * 0.33e-6 + rec.get("sampled_tokens", 0) * 0.84e-6
        async with lock:
            files[eval_name].write(json.dumps(rollout.jsonable(out), ensure_ascii=False) + "\n")
            files[eval_name].flush()
            n_done += 1
            print(f"[{n_done}/{len(jobs)} {time.time() - t0:.0f}s] {eval_name} {out['item_id']} s{sample}: "
                  f"hack={sc.get('hack')} turns={rec['num_turns']}" + (f" ERROR {rec['error']}" if rec["error"] else ""),
                  flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    for f in files.values():
        f.close()
    await h.sandbox.close()


if __name__ == "__main__":
    asyncio.run(main())
