"""SDF-finetune several (arm, run) pairs from ONE process, with checkpoints and resume.

    uv run --no-project --with tinker python experiments/contrastive_sdf/sdf_train.py --tag pilot3 \\
        --arms base,c430,oss430 --features comp,quotes --pairings G,U --resume

Plain next-token loss on raw document text (no chat template, no <DOCTAG> prefix, no pretraining
mix, as in the paper's final recipe). Corpus = union of the run's two universes, shuffled with
``--seed``. Defaults follow the paper (lr 3.5e-5, batch 8, 1 epoch) with 100 warmup steps.

Why this shape (pilot2 never finished): ten separate processes, each holding its own tokenizer and
Python-int token lists, were OOM-killed on the laptop at steps 433-699 of ~750, and nothing but the
midpoint samplers had been saved. Now:
- one process drives every run concurrently (asyncio, one Tinker training client per run). Each
  universe is tokenized once and stored as ``array('i')``, shared by the runs that use it;
- every ``--save-every`` steps each run saves its Tinker training state (weights + optimizer) and a
  sampler, and appends ``{"step", "state", "sampler"}`` to ``<out>/manifest.jsonl``;
- ``--resume`` reloads the last manifest state with its optimizer and skips the steps already done.
  The data order and LR schedule are functions of (docs, seed, args) only, so they are identical;
- a run that fails after retries stops on its own; the others go on, and ``--resume`` picks it up.
- progress (step, loss, $) is printed and appended to ``<out>/metrics.jsonl`` as it goes; the
  totals across runs go to ``results/sdf/<tag>/progress.json``.

On Modal (survives the laptop): see modal_sdf.py. Outputs: ``results/sdf/<tag>/<arm>_<run>/``
{manifest.jsonl, metrics.jsonl, checkpoint.json (final sampler), args.json}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from array import array
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from universes import RUNS, runs_for, universe_id  # noqa: E402

PRICE_TRAIN = 0.737e-6  # $/token, gpt-oss-120b on Tinker
# Arguments that must match on --resume (they fix the data order and the LR schedule).
FIXED = ("lr", "batch", "epochs", "warmup", "max_doc_tokens", "max_docs_per_universe", "seed")


def batches_for(n_docs: int, batch: int, epochs: int, seed: int) -> list[list[int]]:
    """Deterministic data order: the same (n_docs, batch, epochs, seed) always gives the same batches."""
    order: list[int] = []
    rng = random.Random(seed)
    for _ in range(epochs):
        idx = list(range(n_docs))
        rng.shuffle(idx)
        order += idx
    return [order[i:i + batch] for i in range(0, len(order), batch)]


def lr_at(step: int, n_steps: int, lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, n_steps - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * p))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def last_saved(out: Path) -> dict | None:
    m = read_jsonl(out / "manifest.jsonl")
    return max(m, key=lambda r: r["step"]) if m else None


class Corpus:
    """Tokenized universes, loaded lazily and shared across runs."""

    def __init__(self, docs_dir: Path, tok, max_doc_tokens: int, max_docs: int | None) -> None:
        self.docs_dir, self.tok, self.max_doc_tokens, self.max_docs = docs_dir, tok, max_doc_tokens, max_docs
        self.eos = tok.encode("<|endoftext|>", add_special_tokens=False)
        self.cache: dict[str, list[array]] = {}

    def universe(self, uid: str) -> list[array]:
        if uid not in self.cache:
            rows = read_jsonl(self.docs_dir / f"{uid}.jsonl")[: self.max_docs]
            if not rows:
                raise FileNotFoundError(f"no docs for universe {uid} in {self.docs_dir}")
            self.cache[uid] = [array("i", (self.tok.encode(r["text"], add_special_tokens=False) + self.eos)
                                    [: self.max_doc_tokens]) for r in rows]
        return self.cache[uid]

    def run(self, run: str) -> list[array]:
        return [s for auth, style in RUNS[run] for s in self.universe(universe_id(auth, style))]


class Totals:
    def __init__(self, path: Path) -> None:
        self.path, self.runs = path, {}

    def update(self, key: str, **kw) -> None:
        self.runs.setdefault(key, {}).update(kw)
        tot = sum(r.get("cost_usd", 0.0) for r in self.runs.values())
        self.path.write_text(json.dumps({"total_cost_usd": tot, "updated": time.time(), "runs": self.runs}, indent=1))


async def retry(make, what: str, tries: int = 4):
    for attempt in range(tries):
        try:
            return await make()
        except Exception as e:  # transient Tinker errors
            if attempt == tries - 1:
                raise
            print(f"  {what}: {type(e).__name__}: {str(e)[:200]} (retry {attempt + 1})", flush=True)
            await asyncio.sleep(10 * (attempt + 1))


async def train_one(sc, tinker_types, TensorData, corpus: Corpus, arm: str, init: dict, run: str, a, root: Path,
                    totals: Totals) -> str:
    key = f"{arm}_{run}"
    out = root / key
    out.mkdir(parents=True, exist_ok=True)
    args = {k: getattr(a, k) for k in FIXED}
    prev = last_saved(out) if a.resume else None
    if (out / "checkpoint.json").exists() and a.resume:
        print(f"[{key}] already finished", flush=True)
        return "done"
    if prev is None and (out / "manifest.jsonl").exists() and not a.resume:
        sys.exit(f"[{key}] {out} has checkpoints; pass --resume or use a new --tag")
    if (out / "args.json").exists() and a.resume:
        old = json.loads((out / "args.json").read_text())
        bad = {k: (old.get(k), v) for k, v in args.items() if old.get(k) != v}
        if bad:
            sys.exit(f"[{key}] --resume with different fixed args {bad}")
    (out / "args.json").write_text(json.dumps({**args, "arm": arm, "run": run, "init": init}, indent=1))

    seqs = corpus.run(run)
    batches = batches_for(len(seqs), a.batch, a.epochs, a.seed)
    n_steps = len(batches)
    start = prev["step"] if prev else 0
    stop = min(n_steps, a.max_steps) if a.max_steps else n_steps

    if prev:
        tc = await retry(lambda: sc.create_training_client_from_state_with_optimizer_async(prev["state"]),
                         f"[{key}] load state")
        print(f"[{key}] resumed from step {start} ({prev['state']})", flush=True)
    elif init.get("state"):
        tc = await retry(lambda: sc.create_training_client_from_state_async(init["state"]), f"[{key}] load init")
    elif init.get("sampling_only"):
        sys.exit(f"arm {arm} has no training state path; add {{'state': 'tinker://.../weights/...'}} to arms.json")
    else:
        tc = await retry(lambda: sc.create_lora_training_client_async(base_model=a.base_model, rank=32),
                         f"[{key}] new LoRA")

    # Keep metrics for the steps that will not be redone.
    mrows = [r for r in read_jsonl(out / "metrics.jsonl") if r["step"] < start]
    with open(out / "metrics.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in mrows)
    cost = mrows[-1]["cost_usd"] if mrows else 0.0
    print(f"[{key}] {len(seqs)} docs, {sum(map(len, seqs)) / 1e6:.2f}M tok/epoch, steps {start}->{stop} of {n_steps}",
          flush=True)
    totals.update(key, step=start, n_steps=n_steps, cost_usd=cost, status="running")
    mf = open(out / "metrics.jsonl", "a")
    t0 = time.time()

    async def save(step_done: int, final: bool) -> None:
        name = f"csdf-{a.tag}-{key}-{step_done:05d}"
        sampler = (await retry(lambda: tc.save_weights_for_sampler_async(name=name), f"[{key}] save sampler"))
        sampler = (await sampler.result_async()).path
        if final:
            (out / "checkpoint.json").write_text(json.dumps(
                {"sampler": sampler, "arm": arm, "run": run, "steps": step_done, "docs": len(seqs), "args": args},
                indent=2))
            print(f"[{key}] FINISHED, sampler {sampler}", flush=True)
            return
        state = (await retry(lambda: tc.save_state_async(name=name), f"[{key}] save state"))
        state = (await state.result_async()).path
        with open(out / "manifest.jsonl", "a") as f:
            f.write(json.dumps({"step": step_done, "state": state, "sampler": sampler, "time": time.time()}) + "\n")
        print(f"[{key}] saved step {step_done}: {state}", flush=True)

    for step in range(start, stop):
        data = []
        for i in batches[step]:
            s = seqs[i]
            data.append(tinker_types.Datum(
                model_input=tinker_types.ModelInput.from_ints(list(s[:-1])),
                loss_fn_inputs={
                    "target_tokens": TensorData(data=list(s[1:]), dtype="int64", shape=[len(s) - 1]),
                    "weights": TensorData(data=[1.0] * (len(s) - 1), dtype="float32", shape=[len(s) - 1]),
                }))
        n_tok = sum(len(seqs[i]) - 1 for i in batches[step])
        lr = lr_at(step, n_steps, a.lr, a.warmup)

        async def one_step():
            fb = await tc.forward_backward_async(data, loss_fn="cross_entropy")
            op = await tc.optim_step_async(tinker_types.AdamParams(learning_rate=lr))
            r = await fb.result_async()
            await op.result_async()
            return r

        fbr = await retry(one_step, f"[{key}] step {step}")
        cost += n_tok * PRICE_TRAIN
        m = fbr.metrics
        loss = m["loss:sum"] / n_tok if "loss:sum" in m else m.get("loss:mean", float("nan"))
        mf.write(json.dumps({"step": step, "loss": loss, "lr": lr, "tokens": n_tok,
                             "elapsed": time.time() - t0, "cost_usd": cost}) + "\n")
        mf.flush()
        done = step + 1
        totals.update(key, step=done, loss=loss, cost_usd=cost)
        if step % 10 == 0 or done == stop:
            print(f"[{key}] step {done}/{n_steps} loss {loss:.3f} {time.time() - t0:.0f}s ${cost:.2f}", flush=True)
        if done == n_steps:
            await save(done, final=True)
        elif done % a.save_every == 0 or done == stop:
            await save(done, final=False)
    totals.update(key, status="finished" if stop == n_steps else f"stopped at {stop}")
    return "finished" if stop == n_steps else "stopped"


async def main() -> None:
    import tinker
    from tinker import TensorData, types

    from tinker_client import BASE_MODEL, INIT_ARMS, load_env_file

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", required=True, help="comma-separated init arms (tinker_client.INIT_ARMS / arms.json)")
    ap.add_argument("--runs", default=None, help="comma-separated run names; default: all runs for --features x --pairings")
    ap.add_argument("--features", default="comp,quotes")
    ap.add_argument("--pairings", default="G,U")
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-doc-tokens", type=int, default=1536)
    ap.add_argument("--max-docs-per-universe", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=None, help="stop (and save) after this many steps (smoke tests)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--docs-dir", default=str(HERE / "data" / "docs"))
    ap.add_argument("--results-dir", default=os.environ.get("RSRL_SDF_RESULTS", str(HERE / "results")))
    a = ap.parse_args()
    a.base_model = BASE_MODEL
    load_env_file()
    runs = a.runs.split(",") if a.runs else runs_for(a.features.split(","), a.pairings.split(","))
    arms = a.arms.split(",")
    for x in arms:
        if x not in INIT_ARMS:
            sys.exit(f"unknown arm {x}; known: {sorted(INIT_ARMS)}")
    for r in runs:
        if r not in RUNS:
            sys.exit(f"unknown run {r}; known: {sorted(RUNS)}")
    root = Path(a.results_dir) / "sdf" / a.tag
    root.mkdir(parents=True, exist_ok=True)
    sc = tinker.ServiceClient()
    tok = (await sc.create_sampling_client_async(base_model=BASE_MODEL)).get_tokenizer()
    corpus = Corpus(Path(a.docs_dir), tok, a.max_doc_tokens, a.max_docs_per_universe)
    totals = Totals(root / "progress.json")
    pairs = [(x, r) for x in arms for r in runs]
    print(f"{len(pairs)} runs: {pairs}", flush=True)
    res = await asyncio.gather(*[train_one(sc, types, TensorData, corpus, x, INIT_ARMS[x], r, a, root, totals)
                                 for x, r in pairs], return_exceptions=True)
    failed = 0
    for (x, r), v in zip(pairs, res):
        if isinstance(v, BaseException):
            failed += 1
            totals.update(f"{x}_{r}", status=f"failed: {type(v).__name__}: {str(v)[:200]}")
        print(f"{x}_{r}: {v if not isinstance(v, BaseException) else 'FAILED ' + repr(v)[:300]}", flush=True)
    print(f"total ${sum(v.get('cost_usd', 0) for v in totals.runs.values()):.2f}; "
          + ("rerun with --resume to continue failed runs" if failed else "all runs ok"), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
