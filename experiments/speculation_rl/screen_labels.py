"""Build outcome-blind judge inputs from the same preselected sample per episode and arm."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def write_packets(folder, out, kind):
    rows = []
    if kind == "frontier":
        frontier = folder / "frontier_opening"
        wanted = [
            (fam, i, model)
            for fam in ("A", "B", "C")
            for i in range(2)
            for model in (
                "GLM-5.3",
                "GLM-5.3-Flash",
                "Kimi-K2.6",
                "aghyad430",
                "control430",
                "untrained",
            )
        ]
        for fam, i, arm in wanted:
            if arm in ("aghyad430", "control430", "untrained"):
                f = folder / f"{fam}-{i:02}-identity-{arm}.jsonl"
                r = next(
                    r
                    for r in map(json.loads, f.read_text(encoding="utf8").splitlines())
                    if r["sample"] == 0
                )
            else:
                f = frontier / f"{arm}-{fam}-{i:02}-identity.json"
                r = json.loads(f.read_text(encoding="utf8"))
            rows.append((f.name, r))
    else:
        for fam in "ABCEF":
            for i in range(50):
                for arm in ("aghyad430", "control430", "untrained"):
                    f = folder / f"{fam}-{i:02}-identity-{arm}.jsonl"
                    r = next(
                        r
                        for r in map(
                            json.loads, f.read_text(encoding="utf8").splitlines()
                        )
                        if r["sample"] == 0
                    )
                    rows.append((f.name, r))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).parent / "validation/REVIEW.txt", out / "REVIEW.txt")
    packets = []
    mapping = []
    for fname, r in rows:
        turn = r["turns"][-1]
        ident = hashlib.sha256((kind + "/v1/" + fname).encode()).hexdigest()[:16]
        missing = not turn["reasoning"].strip()
        mapping.append(
            dict(
                id=ident,
                source=fname,
                sample=0,
                family=r["family"],
                arm=r["arm"],
                index=r["index"],
                missing_reasoning=missing,
                correct=bool(turn["grade"].get("optimal", r["reward"] == 1)),
                reward=r["reward"],
                cutoff=turn["cutoff"],
            )
        )
        if not missing:
            packets.append(
                dict(
                    id=ident,
                    prompt=turn["prompt"],
                    reasoning=turn["reasoning"],
                    reply=turn["reply"] if "reply" in turn else turn["final"],
                )
            )
    for name, items in (("packets", packets), ("unblinding", mapping)):
        (out / (name + ".jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in sorted(items, key=lambda r: r["id"])),
            encoding="utf8",
        )
    print("eligible", len(packets), "of", len(mapping))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("kind", choices=["frontier", "screen"])
    a = p.parse_args()
    write_packets(a.folder, a.out, a.kind)
