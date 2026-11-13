"""Compare fresh, jointly observed episodes across six arms; keep both blind judges."""

import json, random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from judge import MODELS


def analyze(folder):
    rows = [
        json.loads(line)
        for line in (folder / "unblinding.jsonl").read_text().splitlines()
    ]
    data = defaultdict(dict)
    for r in rows:
        files = [folder / "span_judges" / m / (r["id"] + ".json") for m in MODELS]
        if not all(f.exists() for f in files):
            continue
        data[r["arm"]][r["family"], r["index"]] = [
            json.loads(f.read_text())["evidence"]["unseen_grader"]["present"]
            for f in files
        ]
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    rng = random.Random(816250)
    stats = {}
    for arm, values in data.items():
        stats[arm] = dict(
            observed=len(values),
            common=len(common),
            positives=[sum(values[k][j] for k in common) for j in range(2)],
        )
    diff = {}
    for model in ("GLM-5.3", "GLM-5.3-Flash"):
        for ref in ("Kimi-K2.6", "control430", "aghyad430", "untrained"):
            for j, name in enumerate(MODELS):
                # Resample tasks within each of the three prespecified prompt families.
                strata = [
                    [
                        int(data[model][k][j]) - int(data[ref][k][j])
                        for k in common
                        if k[0] == family
                    ]
                    for family in "ABC"
                ]
                bootstrap = sorted(
                    mean(
                        [
                            y
                            for group in strata
                            for y in rng.choices(group, k=len(group))
                        ]
                    )
                    for _ in range(4000)
                )
                diff[model + " minus " + ref + " " + name] = dict(
                    raw=mean([y for s in strata for y in s]),
                    paired_ci95=[bootstrap[99], bootstrap[3899]],
                )
    return dict(
        common_tasks=common,
        per_arm=stats,
        paired=diff,
        unit="One preselected completion per arm on each episode; task bootstrap within family. Single sampling seed.",
    )


if __name__ == "__main__":
    print(json.dumps(analyze(Path(__import__("sys").argv[1])), indent=2))
