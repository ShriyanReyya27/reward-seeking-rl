"""Paired P1 framing comparisons and P2 instruction-gradient checks."""

import json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
import random


def summary(folder):
    grouped = defaultdict(dict)
    for p in folder.glob("*.jsonl"):
        rows = list(map(json.loads, p.read_text(encoding="utf8").splitlines()))
        if len(rows) != 8:
            raise ValueError("Incomplete group")
        r = rows[0]
        key = (r["phase"], r["family"], r["variant"], r["arm"])
        rewards = [x["reward"] for x in rows]
        grouped[key][r["index"]] = dict(
            paid=mean(rewards),
            mixed=int(min(rewards) != max(rewards)),
            cutoff=mean(x["cutoff"] for x in rows),
            binding=r["binding"],
        )
    rng = random.Random(816200)
    table = []
    contrasts = []
    for key, groups in sorted(grouped.items()):
        phase, fam, v, arm = key
        item = dict(phase=phase, family=fam, variant=v, arm=arm, n=len(groups))
        item.update(
            {
                k: mean(r[k] for r in groups.values())
                for k in ("paid", "mixed", "cutoff")
            }
        )
        item["p2_instruction_in_target"] = (
            0.70 <= item["paid"] <= 0.85
            if phase == "P2" and v == "instruction"
            else None
        )
        table.append(item)
        if phase == "P1" and v == "grading":
            down = grouped["P1", fam, "downstream", arm]
            ids = sorted(set(groups) & set(down))
            for k in ("paid", "mixed"):
                deltas = [groups[i][k] - down[i][k] for i in ids]
                boot = sorted(
                    mean(rng.choices(deltas, k=len(ids))) for _ in range(4000)
                )
                contrasts.append(
                    dict(
                        family=fam,
                        arm=arm,
                        quantity=k,
                        groups=len(ids),
                        grading_minus_downstream=mean(deltas),
                        interval=[boot[99], boot[3899]],
                    )
                )
    return dict(table=table, paired_framing=contrasts)


if __name__ == "__main__":
    print(json.dumps(summary(Path(sys.argv[1])), indent=2))
