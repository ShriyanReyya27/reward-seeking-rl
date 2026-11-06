"""Learning against evidence rather than steps, for a fixed-rule rl.py run.

    uv run python experiments/hidden_preference/evidence_curves.py cued_v1

GRPO skips groups with no reward variance, so an arm whose samples all score the same gets no gradient
from them. This counts, per arm and family, the informative groups (reward variance > 0) and the
rewarded samples inside them, cumulated over training steps. If accuracy lines up across arms when
plotted against this count, the step gap comes from when evidence arrives; if one arm still rises
faster, it gets more out of the same evidence.

Writes results/<run>/figure/evidence.csv (one row per arm, family, training step: train accuracy
and cumulative counts up to and including that step) and prints the evidence needed to reach set
train-accuracy levels, on a centred 3-step average.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("c430", "oss430")
LEVELS = (0.25, 0.5, 0.75, 0.9)


def rollouts(run, arm):
    rows = []
    for p in sorted(glob.glob(str(HERE / "results" / run / arm / "rollouts_from*.jsonl.gz"))):
        text = zlib.decompressobj(31).decompress(Path(p).read_bytes()).decode(errors="ignore")  # tolerates a killed writer
        for line in text.split("\n"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main(run):
    out = HERE / "results" / run / "figure"
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for arm in ARMS:
        rows = rollouts(run, arm)
        steps = sorted({r["step"] for r in rows})
        cfg = json.loads((HERE / "results" / run / arm / "config.json").read_text())
        assert len(rows) == len(steps) * int(cfg["batch"]) * int(cfg["group"]), (arm, len(rows))
        groups = defaultdict(list)
        for r in rows:
            groups[(r["family"], r["step"], r["task_id"])].append(r["reward"])
        for fam in sorted({r["family"] for r in rows}):
            cum_g = cum_pos = 0
            for s in steps:
                g = [v for (f, st, _), v in groups.items() if f == fam and st == s]
                inf = [v for v in g if max(v) > min(v)]
                cum_g += len(inf)
                cum_pos += sum(sum(v) for v in inf)
                table.append({"arm": arm, "family": fam, "step": s, "groups": len(g), "informative_groups": len(inf),
                              "cum_informative_groups": cum_g, "cum_rewarded_in_informative": int(cum_pos),
                              "train_accuracy_pct": round(100 * sum(sum(v) for v in g) / max(1, sum(len(v) for v in g)), 1)})
    for arm in ARMS:  # centred 3-step average of train accuracy, for plotting (fewer at the ends)
        for fam in {t["family"] for t in table}:
            t = [x for x in table if x["arm"] == arm and x["family"] == fam]
            for i, x in enumerate(t):
                w = t[max(0, i - 1):i + 2]
                x["train_accuracy_3step_pct"] = round(sum(y["train_accuracy_pct"] for y in w) / len(w), 1)
    with open(out / "evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    for fam in sorted({t["family"] for t in table}):
        print(f"\n== {fam}: evidence when the 3-step train accuracy first reaches a level")
        print("level".ljust(8) + "".join(f"{a}: step, inf. groups, rewarded".rjust(36) for a in ARMS))
        for lv in LEVELS:
            line = f"{int(100 * lv)}%".ljust(8)
            for arm in ARMS:
                t = [x for x in table if x["arm"] == arm and x["family"] == fam]
                hit = next((t[i] for i in range(1, len(t) - 1)
                            if sum(t[j]["train_accuracy_pct"] for j in (i - 1, i, i + 1)) / 3 >= 100 * lv), None)
                line += (f"{hit['step']}, {hit['cum_informative_groups']}, {hit['cum_rewarded_in_informative']}" if hit else "not reached").rjust(36)
            print(line)


if __name__ == "__main__":
    main(sys.argv[1])
