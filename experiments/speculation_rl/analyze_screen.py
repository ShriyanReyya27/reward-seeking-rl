"""Summarize completed S1 groups; bootstrap paired episodes, never individual samples."""

from collections import defaultdict
from pathlib import Path
from statistics import mean
import json, random, sys
from tasks import FAMILIES, VERSIONS


def summarize(folder):
    data = defaultdict(dict)
    for path in folder.glob("*.jsonl"):
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf8").splitlines()
        ]
        if not rows or "turns" not in rows[0]:
            continue
        r = rows[0]
        assert len(rows) == 8 and {x["sample"] for x in rows} == set(range(8))
        for x in rows:
            assert (x["family"], x["version"], x["seed"], x["arm"]) == (
                r["family"],
                r["version"],
                r["seed"],
                r["arm"],
            )
        correct = [
            row["turns"][-1]["grade"].get("optimal", row["reward"] == 1) for row in rows
        ]
        grades = [row["turns"][-1]["grade"] for row in rows]
        result = dict(
            paid=mean(x["reward"] for x in rows),
            true_score=mean(x["true_score"] for x in rows),
            success=mean(correct),
            mixed=int(len(set(correct)) > 1),
            cutoff=mean(any(t["cutoff"] for t in x["turns"]) for x in rows),
            tool_failures=mean(
                any(t["tool_failure"] for t in x["turns"]) for x in rows
            ),
            output_tokens=mean(sum(t["tokens"] for t in x["turns"]) for x in rows),
        )
        if r["family"] in ("A", "F"):
            result["visible_pass_hidden_fail"] = mean(
                g["visible_pass"] and not g["hidden_pass"] for g in grades
            )
        key = (r["family"], r["version"], r["arm"])
        if r["seed"] in data[key]:
            raise ValueError("Duplicate episode")
        data[key][r["seed"]] = result
    table = []
    for (family, version, arm), episodes in sorted(data.items()):
        table.append(
            dict(
                family=family,
                version=version,
                arm=arm,
                groups=len(episodes),
                **{
                    k: mean(row[k] for row in episodes.values())
                    for k in next(iter(episodes.values()))
                },
            )
        )
    comparisons = []
    rng = random.Random(121126)
    for family in FAMILIES:
        for version in VERSIONS:
            a = data[family, version, "aghyad430"]
            b = data[family, version, "control430"]
            ids = sorted(set(a) & set(b))
            if not ids:
                continue
            for metric in ("paid", "mixed", "success"):
                diffs = [a[i][metric] - b[i][metric] for i in ids]
                boot = sorted(mean(rng.choices(diffs, k=len(ids))) for _ in range(2000))
                comparisons.append(
                    dict(
                        family=family,
                        version=version,
                        metric=metric,
                        groups=len(ids),
                        difference=mean(diffs),
                        ci95=[boot[49], boot[1949]] if len(ids) > 1 else None,
                    )
                )
    return dict(table=table, paired_comparisons=comparisons)


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
