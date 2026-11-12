"""Build the coding problem set from DeepCoder's TACO-verified split, binned by TACO's own difficulty labels.

    python build_coding.py --data <dir with deepcoder/taco_*.parquet, deepcoder/lcbv5_*.parquet, taco/meta.parquet>

Keeps stdin/stdout problems with >= 5 tests, drops anything that may overlap LiveCodeBench (TACO date on or after
2023-05-01, when LiveCodeBench's problem window starts, or statement text matching a DeepCoder LCB problem), and for
each bin shuffles the pool (seed 0). Problems with > 15 tests keep the 15 with the longest inputs; problems whose kept
tests all expect the same output are dropped (a constant answer would pass).

Step 1 (no --verified): writes coding_candidates.jsonl.gz, the first 300 of each bin's shuffled pool.
Step 2 (--verified grades_verify.jsonl): keeps candidates where one of up to 3 official solutions passes every kept
test, in the same order, and assigns the first 100 per bin to the screen and the next 100 to the held-out set.
Writes coding_problems.jsonl.gz (tests included) and coding_bins.json.
"""

import argparse
import glob
import gzip
import json
import random
import re

import pandas as pd

from common import OUT

BINS = {"B1_medium": "MEDIUM", "B2_medium_hard": "MEDIUM_HARD", "B3_hard": "HARD", "B4_very_hard": "VERY_HARD"}
INSTRUCTION = ("Write a complete Python 3 solution that reads from standard input and writes to standard output. Put the "
               "code in a single ```python block. No tools are available; answer directly.")
N_SCREEN, N_HELDOUT, MAX_TESTS, N_CANDIDATES = 100, 100, 15, 300


def squash(s):
    return re.sub(r"\s+", " ", s or "").strip()


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()[:400]


def as_text(x):
    return "\n".join(map(str, x)) if isinstance(x, list) else str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--verified", help="grades_verify.jsonl from `coding.py verify`")
    args = p.parse_args()
    if args.verified:
        return assign(args.verified)
    meta = pd.read_parquet(f"{args.data}/taco/meta.parquet")
    meta["key"] = meta.question.map(norm)
    # a statement listed twice with different labels is ambiguous: drop it
    labels = meta.groupby("key").difficulty.nunique()
    meta = meta[meta.key.map(labels) == 1].drop_duplicates("key")
    dc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{args.data}/deepcoder/taco_*.parquet"))])
    dc["key"] = dc.problem.map(norm)
    lcb = pd.concat([pd.read_parquet(f, columns=["problem"]) for f in glob.glob(f"{args.data}/deepcoder/lcbv5_*.parquet")])
    lcb_keys = set(lcb.problem.map(lambda s: norm(s)[:200]))
    m = dc.merge(meta, on="key", how="inner")
    log = {"deepcoder_taco": len(dc), "with_unambiguous_label": len(m)}
    rows = []
    for _, r in m.iterrows():
        t = json.loads(r.tests)
        if not isinstance(t, dict) or t.get("fn_name") or len(t.get("inputs", [])) < 5:
            continue
        date = r.date if isinstance(r.date, str) else ""  # most TACO problems are undated (NaN)
        if date >= "2023-05-01" or r.key[:200] in lcb_keys:
            log["dropped_lcb_overlap"] = log.get("dropped_lcb_overlap", 0) + 1
            continue
        tests = [(as_text(i), as_text(o)) for i, o in zip(t["inputs"], t["outputs"])]
        n_all = len(tests)
        if len(tests) > MAX_TESTS:
            tests = sorted(tests, key=lambda io: -len(io[0]))[:MAX_TESTS]
        if len({squash(o) for _, o in tests}) < 2:
            log["dropped_constant_output"] = log.get("dropped_constant_output", 0) + 1
            continue
        statement = squash(r.problem)
        all_tests = [(as_text(i), as_text(o)) for i, o in zip(t["inputs"], t["outputs"])]
        examples = [o for i, o in all_tests if squash(i) and squash(i) in statement]  # tests shown in the statement
        sols = json.loads(r.solutions) if isinstance(r.solutions, str) else list(r.solutions if r.solutions is not None else [])
        rows.append({"difficulty": r.difficulty, "source": r.source, "url": r.url, "date": date or None,
                     "time_limit": r.time_limit, "memory_limit": r.memory_limit, "problem": r.problem,
                     "tests": tests, "n_tests_total": n_all, "example_outputs": examples, "solutions": sols[:3]})
    log["stdio_ge5_tests_no_overlap"] = len(rows)
    rng = random.Random(0)
    out, bins = [], {}
    log["candidate_pool"] = len(rows)
    for b, label in BINS.items():
        pool = [r for r in rows if r["difficulty"] == label]
        pool.sort(key=lambda r: r["url"] + r["problem"][:50])
        rng.shuffle(pool)
        bins[b] = {"taco_difficulty": label, "available": len(pool),
                   "sources": pd.Series([r["source"] for r in pool]).value_counts().to_dict()}
        for i, r in enumerate(pool[:N_CANDIDATES]):
            out.append({"id": f"{b}/{i:03d}", "bin": b, "prompt": f"{r['problem'].rstrip()}\n\n{INSTRUCTION}", **r})
    with gzip.open(OUT / "coding_candidates.jsonl.gz", "wt") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    (OUT / "coding_bins.json").write_text(json.dumps({"filter": log, "bins": bins}, indent=1))
    print(json.dumps({"filter": log, "bins": bins}, indent=1))


def assign(verified):
    ok = {}
    for g in map(json.loads, open(verified)):
        pid, k = g["key"].rsplit("|", 1)
        if g["reward"] and pid not in ok:
            ok[pid] = int(k)
    with gzip.open(OUT / "coding_candidates.jsonl.gz", "rt") as f:
        cands = [json.loads(line) for line in f]
    info = json.loads((OUT / "coding_bins.json").read_text())
    out = []
    for b in BINS:
        pool = [c for c in cands if c["bin"] == b]
        good = [c for c in pool if c["id"] in ok]
        for i, c in enumerate(good[:N_SCREEN + N_HELDOUT]):
            c["split"] = "screen" if i < N_SCREEN else "heldout"
            c["verified_solution"] = ok[c["id"]]
            out.append(c)
        info["bins"][b].update(candidates=len(pool), official_solution_passes=len(good),
                               screen=min(N_SCREEN, len(good)), heldout=max(0, min(N_HELDOUT, len(good) - N_SCREEN)),
                               rl_train_pool_after_screen_and_heldout=max(0, info["bins"][b]["available"] - len(pool))
                               + max(0, len(good) - N_SCREEN - N_HELDOUT))
    with gzip.open(OUT / "coding_problems.jsonl.gz", "wt") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    (OUT / "coding_bins.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info["bins"], indent=1))


if __name__ == "__main__":
    main()
