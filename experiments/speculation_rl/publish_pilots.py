"""Verify and archive the fresh frontier block, both framing pilots and blind labels."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analyze_confirmation import analyze
from analyze_framing import summary
from judge import MODELS


def archive(path, rows):
    with gzip.open(path, "xt", encoding="utf8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def write_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2)


def labels(source, dest):
    mapping = [
        json.loads(line)
        for line in (source / "unblinding.jsonl").read_text(encoding="utf8").splitlines()
    ]
    packets = {
        r["id"]: r
        for r in map(
            json.loads, (source / "packets.jsonl").read_text(encoding="utf8").splitlines()
        )
    }
    assert len({r["id"] for r in mapping}) == len(mapping)
    for row in mapping:
        evidence = {}
        for model in MODELS:
            path = source / "span_judges" / model / (row["id"] + ".json")
            if path.exists():
                entry = json.loads(path.read_text(encoding="utf8"))
                assert entry["id"] == row["id"]
                assert all(
                    flag["quote"] in packets[row["id"]]["reasoning"]
                    for flag in entry["evidence"].values()
                    if flag["quote"]
                )
                evidence[model] = entry
        row["judges"] = evidence
    archive(dest / "trace_labels.jsonl.gz", mapping)
    stats = json.loads((source / "span_judges/summary.json").read_text())
    write_json(dest / "label_agreement.json", stats)


def frontier(root, dest):
    source = root / "frontier_confirmation"
    design = json.loads((source / "design.json").read_text())
    evidence = json.loads((source / "UNRESOLVED_REQUESTS.json").read_text())
    lost = {key.rsplit("-t", 1)[0] for key in evidence["reserved_requests"]}
    write_json(dest / "design.json", design)
    write_json(dest / "unresolved.json", evidence)
    rows = []
    missing = set()
    cells = defaultdict(list)
    for arm in [*design["models"], "aghyad430", "control430", "untrained"]:
        for fam in design["families"]:
            for version in design["versions"]:
                for i in design["indices"]:
                    key = f"{arm}-{fam}-{i:02}-{version}"
                    if arm in design["models"]:
                        path = source / (key + ".json")
                        if key in lost:
                            assert not path.exists()
                            missing.add(key)
                            continue
                        r = json.loads(path.read_text(encoding="utf8"))
                        rows.append(r)
                    else:
                        path = root / f"{fam}-{i:02}-{version}-{arm}.jsonl"
                        if not path.exists():
                            requests = json.loads((root / "UNRESOLVED_REQUESTS.json").read_text())
                            assert any(t.startswith(path.stem + "-t") for t in requests["reserved_requests"])
                            continue
                        with path.open(encoding="utf8") as stream:
                            r = json.loads(next(stream))
                        assert r["sample"] == 0
                    assert (r["arm"], r["family"], r["index"], r["version"]) == (arm, fam, i, version)
                    assert r["reward"] == r["true_score"]
                    assert len(r["turns"]) == (3 if fam == "B" else 1)
                    cells[arm, fam, version].append(r)
    assert missing == lost
    archive(dest / "frontier_episodes.jsonl.gz", rows)
    table = []
    for (arm, fam, version), group in sorted(cells.items()):
        table.append(dict(
            arm=arm, family=fam, version=version, episodes=len(group),
            paid=mean(r["reward"] for r in group), true_score=mean(r["true_score"] for r in group),
            cutoff=mean(any(t["cutoff"] for t in r["turns"]) for r in group),
            tokens=mean(sum(t["tokens"] for t in r["turns"]) for r in group),
        ))
    with (dest / "per_condition.csv").open("x", newline="", encoding="utf8") as stream:
        w = csv.DictWriter(stream, table[0].keys())
        w.writeheader()
        w.writerows(table)
    labelled = root / "confirmation_labels"
    labels(labelled, dest)
    write_json(dest / "paired_judge_comparisons.json", analyze(labelled))


def pilots(root, dest):
    source = root / "framing_gradient"
    jobs = json.loads((source / "jobs.json").read_text(encoding="utf8"))
    rows = []
    for job in jobs:
        for arm in ("aghyad430", "control430", "untrained"):
            name = f"{job['phase']}-{job['family']}-{job['index']:02}-{job['variant']}-{arm}.jsonl"
            group = list(map(json.loads, (source / name).read_text(encoding="utf8").splitlines()))
            assert [r["sample"] for r in group] == list(range(8))
            assert all(r["arm"] == arm and r["reward"] == r["true_score"] for r in group)
            assert all(all(r[key] == val for key, val in job.items()) for r in group)
            rows.extend(group)
    archive(dest / "framing_gradient.jsonl.gz", rows)
    write_json(dest / "jobs.json", jobs)
    write_json(dest / "frame_and_gradient.json", summary(source))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("frontier", "pilots", "screen-labels"))
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    a.dest.mkdir(exist_ok=False, parents=True)
    if a.stage == "frontier":
        frontier(a.root, a.dest)
    elif a.stage == "pilots":
        pilots(a.root, a.dest)
    else:
        labels(a.root / "screen_labels", a.dest)
    (a.dest / "SHA256SUMS").write_text("".join(
        hashlib.sha256(f.read_bytes()).hexdigest() + "  " + f.name + "\n"
        for f in sorted(a.dest.iterdir())
    ), encoding="utf8")
