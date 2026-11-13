"""Prepare a reproducible, blinded validation set from previously collected rollouts."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

LABELS = (
    "visible_grader",
    "unseen_grader",
    "evaluation_awareness",
    "training_awareness",
    "none",
)


def lines(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def prepare(results, out):
    # Stratify by provenance before looking at either judge's decisions.
    prior = results / "persistence_mastermind_20261107/analysis"
    key = {r["id"]: r for r in lines(prior / "unblinding_key.jsonl")}
    groups = {}
    for row in lines(prior / "blinded_packets.jsonl"):
        source = key[row["id"]]
        if not source["snapshot"].endswith("step0") or not row["reasoning"].strip():
            continue
        group = (source["snapshot"], source["environment"])
        groups.setdefault(group, []).append((source["response_id"], row))
    rng = random.Random(121126)
    selected = []
    for group, items in sorted(groups.items()):
        selected.extend((group, item) for item in rng.sample(items, min(8, len(items))))
    extra = []
    for row in lines(results / "phase1/toby_diagnostics/responses.jsonl"):
        if row["arm"] == "base" and row["reasoning"].strip():
            extra.append(
                (
                    ("base", row["variant"]),
                    (
                        row["id"],
                        dict(
                            prompt=row["task"]["prompt"],
                            reasoning=row["reasoning"],
                            final=row["final"],
                        ),
                    ),
                )
            )
    rng.shuffle(extra)
    selected += extra[: 100 - len(selected)]
    assert len(selected) == 100
    rng.shuffle(selected)
    out.mkdir(parents=True, exist_ok=True)
    with (
        (out / "packets.jsonl").open("x", encoding="utf-8") as packets,
        (out / "unblind.jsonl").open("x", encoding="utf-8") as mapping,
        (out / "human.csv").open("x", newline="") as form,
    ):
        writer = csv.writer(form)
        writer.writerow(["id", *LABELS, "evidence_or_notes"])
        for group, (source, row) in selected:
            code = hashlib.sha256(
                ("spec-validation-v1/" + source).encode()
            ).hexdigest()[:12]
            packets.write(
                json.dumps(
                    dict(
                        id=code,
                        prompt=row["prompt"],
                        reasoning=row["reasoning"],
                        reply=row["final"],
                    )
                )
                + "\n"
            )
            mapping.write(
                json.dumps(
                    dict(id=code, arm=group[0], family=group[1], source_id=source)
                )
                + "\n"
            )
            writer.writerow([code] + [""] * 6)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    prepare(a.results, a.out)
