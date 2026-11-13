"""Verify complete paired GPT screen groups, publish summaries and split raw archives."""

import argparse, csv, gzip, hashlib, json
from pathlib import Path
from collections import Counter
from tasks import FAMILIES, VERSIONS
from analyze_screen import summarize


def report(folder, dest, indices):
    dest.mkdir(exist_ok=False, parents=True)
    pool = [
        r
        for r in map(
            json.loads,
            (Path(__file__).parent / "episodes.jsonl")
            .read_text(encoding="utf8")
            .splitlines(),
        )
        if r["split"] == "screen"
    ]
    audit = Counter()
    for f in FAMILIES:
        for arm in ("aghyad430", "control430", "untrained"):
            with gzip.open(
                dest / f"{f}-{arm}.jsonl.gz", "wt", encoding="utf8", newline="\n"
            ) as archive:
                for i in indices:
                    item = next(r for r in pool if r["family"] == f and r["index"] == i)
                    for v in VERSIONS:
                        path = folder / f"{f}-{i:02}-{v}-{arm}.jsonl"
                        rows = [
                            json.loads(l)
                            for l in path.read_text(encoding="utf8").splitlines()
                        ]
                        assert sorted(r["sample"] for r in rows) == list(range(8))
                        for r in rows:
                            assert (
                                r["seed"],
                                r["arm"],
                                r["index"],
                                r["family"],
                                r["version"],
                            ) == (item["seed"], arm, i, f, v)
                            assert r["prompt"] == item["versions"][v]
                            assert len(r["turns"]) == (3 if f == "B" else 1)
                            assert (
                                r["true_score"]
                                == r["reward"]
                                == r["turns"][-1]["grade"]["reward"]
                            )
                            assert (
                                r["model_path"] is None
                                if arm == "untrained"
                                else bool(r["model_path"])
                            )
                            audit["episodes"] += 1
                            audit["turns"] += len(r["turns"])
                            audit["missing_reasoning"] += sum(
                                not t["reasoning"].strip() for t in r["turns"]
                            )
                            audit["empty_final"] += sum(
                                not t["reply"].strip() for t in r["turns"]
                            )
                            audit["turn_cutoffs"] += sum(
                                t["cutoff"] for t in r["turns"]
                            )
                            archive.write(json.dumps(r) + "\n")
    stats = summarize(folder, indices)
    # This snapshot must correspond to exactly the requested indices.
    assert (
        all(r["groups"] == len(indices) for r in stats["table"])
        and len(stats["table"]) == 75
    )
    with (dest / "per_condition.csv").open("w", newline="") as f:
        keys = list(dict.fromkeys(k for s in stats["table"] for k in s))
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(stats["table"])
    (dest / "paired_differences.json").write_text(
        json.dumps(stats["paired_comparisons"], indent=2)
    )
    (dest / "audit.json").write_text(json.dumps(audit, indent=2))
    (dest / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
            for p in sorted(dest.iterdir())
        )
    )
    return stats, audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("n", type=int)
    a = p.parse_args()
    s, r = report(a.folder, a.dest, range(a.n))
    print(dict(r))
