"""Blinded judge batches for the strategies-by-step question. Writes judging/ (instructions unchanged from ../judging/instructions.md).

    uv run python experiments/hidden_preference/meeting/b_reasoning/strategies/prepare.py
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from corpus_st import ARMS, RUNS, load_all

HERE = Path(__file__).resolve().parent
OUT = HERE / "judging"
SEED = 20261109
STEPS = [0, 1, 2, 3, 4, 5, 6, 7] + list(range(8, 35, 2))
N = 10
N_BATCHES = 16
SONNET_PER_STRATUM = 8


def request_text(info):
    return f"Give an integer from {info['lo']} to {info['hi']} that is {info['cons']}."


def entry_text(eid, r):
    return (f"=== ENTRY id={eid} ===\nREQUEST: {request_text(r['info'])}\nFINAL ANSWER: {r['answer'] or '(none)'}\n"
            f"REASONING:\n{r['analysis']}\n=== END ENTRY id={eid} ===\n\n")


def main():
    rows, takeoffs, audit = load_all()
    rng = random.Random(SEED)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["run"], r["arm"], r["step"])].append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE.parent / "judging" / "instructions.md", OUT / "instructions.md")
    entries = []
    for run in RUNS:
        for arm in ARMS:
            for s in STEPS:
                xs = cells[(run, arm, s)]
                by_req = defaultdict(list)
                for r in xs:
                    by_req[r["base_id"]].append(r)
                reqs = sorted(by_req)
                rng.shuffle(reqs)
                for q in reqs:
                    rng.shuffle(by_req[q])
                picked = []
                while len(picked) < N and any(by_req.values()):
                    for q in reqs:
                        if by_req[q] and len(picked) < N:
                            picked.append(by_req[q].pop())
                assert len(picked) == N, (run, arm, s, len(picked))
                entries += picked
    rng.shuffle(entries)
    ids = rng.sample(range(100_000, 1_000_000), len(entries))
    key, texts = {}, []
    for eid, r in zip(ids, entries):
        t = takeoffs[f"{r['run']}:{r['arm']}"]
        key[eid] = {"run": r["run"], "arm": r["arm"], "step": r["step"], "base_id": r["base_id"], "task_id": r["task_id"],
                    "cls": r["cls"], "answer": r["answer"], "n_words": r["n_words"], "takeoff": t, "rel": r["step"] - t}
        texts.append((eid, entry_text(eid, r)))
    size = sum(len(t) for _, t in texts) / N_BATCHES
    batches, cur, s = [], [], 0
    for eid, t in texts:
        if cur and s + len(t) > size * 1.04 and len(batches) < N_BATCHES - 1:
            batches.append(cur)
            cur, s = [], 0
        cur.append((eid, t))
        s += len(t)
    batches.append(cur)
    for b, items in enumerate(batches):
        p = OUT / f"batch_{b:02d}.txt"
        p.write_text("".join(t for _, t in items))
    (OUT / "key.json").write_text(json.dumps(key))
    strat = defaultdict(list)
    for eid, t in texts:
        k = key[eid]
        strat[(k["run"], k["arm"], k["rel"] < 0)].append((eid, t))
    check = []
    for k in sorted(strat):
        check += rng.sample(strat[k], min(SONNET_PER_STRATUM, len(strat[k])))
    rng.shuffle(check)
    for c in range(3):
        (OUT / f"check_{c}.txt").write_text("".join(t for _, t in check[c::3]))
    (OUT / "check_ids.json").write_text(json.dumps([e for e, _ in check]))
    (OUT / "manifest.json").write_text(json.dumps(
        {"batches": {f"batch_{b:02d}": [e for e, _ in items] for b, items in enumerate(batches)},
         "checks": {f"check_{c}": [e for e, _ in check[c::3]] for c in range(3)}}))
    print(len(entries), "entries", len(batches), "batches, check", len(check))
    for b, items in enumerate(batches):
        print(f"batch_{b:02d}", len(items), f"{sum(len(t) for _, t in items) / 1e3:.0f}k", sum(t.count(chr(10)) for _, t in items), "lines")
    for c in range(3):
        print("check", c, len(check[c::3]), f"{sum(len(t) for _, t in check[c::3]) / 1e3:.0f}k")


if __name__ == "__main__":
    main()
