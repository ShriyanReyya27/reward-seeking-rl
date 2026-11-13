"""Read-only progress audit. Count distinct groups, episodes and model replies separately."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def status(folder, ledger):
    groups = Counter()
    turns = Counter()
    cutoffs = Counter()
    arm_turns = Counter()
    samples = 0
    for path in folder.glob("*.raw.json"):
        record = json.loads(path.read_text(encoding="utf8"))
        turns[path.name[0]] += len(record["raw"])
    cells = defaultdict(list)
    for path in folder.glob("*.jsonl"):
        with path.open(encoding="utf8") as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != 8:
            raise ValueError(f"Unexpected group size in {path.name}")
        r = rows[0]
        groups[r["family"]] += 1
        samples += len(rows)
        cells[(r["family"], r["arm"], r["version"])].append(
            (
                r["index"],
                sum(
                    x["turns"][-1]["grade"].get("optimal", x["reward"] == 1)
                    for x in rows
                )
                / 8,
            )
        )
        for row in rows:
            arm_turns[r["arm"]] += len(row["turns"])
            cutoffs[r["arm"]] += sum(t["cutoff"] for t in row["turns"])
    latest = {
        x["id"]: x
        for x in map(json.loads, ledger.read_text(encoding="utf8").splitlines())
    }
    cost = Counter()
    for k, x in latest.items():
        if k.startswith("speculation-v1/"):
            cost[x["event"]] += x["usd"]
    return dict(
        groups=groups,
        graded_episodes=samples,
        replies_received=turns,
        settled_sampling_usd=round(cost["settle"], 4),
        pending_sampling_reservations_usd=round(cost["reserve"], 4),
        campaign_committed_usd=round(sum(x["usd"] for x in latest.values()), 3),
        cutoffs=cutoffs,
        graded_turns=arm_turns,
        identity_groups={
            str((f, a)): g for (f, a, v), g in cells.items() if v == "identity"
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("ledger", type=Path)
    a = p.parse_args()
    print(json.dumps(status(a.out, a.ledger), indent=2))
