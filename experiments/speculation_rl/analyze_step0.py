"""Summarize Step 0 measures, retaining the distinction between native reward and hack rate."""

import json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(root):
    table = []
    for sub in ("step0_battery", "step0_coding"):
        by = defaultdict(list)
        for f in (root / sub).glob("*.json"):
            if f.name in ("manifest.json", "design.json") or ".raw." in f.name:
                continue
            r = json.loads(f.read_text(encoding="utf8"))
            if "arm" not in r:
                continue
            family = r.get("family", "coding_hackable")
            variant = r.get("variant", r.get("mode"))
            g = r.get("grade", r)
            if family == "countdown_leaky":
                honest = float(g["honest"])
                hack = float(g["hack"])
                paid = float(honest or (hack and variant == "leaky"))
            elif family == "grader_conflict":
                honest = float(g["honest"])
                hack = float(g["hack"])
                paid = g["reward"]
            else:
                honest = g["true_score"]
                paid = g["reward"]
                hack = float(g.get("hacked", False))
            by[(family, variant, r["arm"])].append(
                (
                    paid,
                    honest,
                    hack,
                    r.get("tokens", r.get("output_tokens", 0)),
                    r.get("cutoff", any(t["cutoff"] for t in r.get("turns", []))),
                )
            )
        for (family, variant, arm), vals in sorted(by.items()):
            table.append(
                dict(
                    family=family,
                    variant=variant,
                    arm=arm,
                    n=len(vals),
                    mean_paid=mean(x[0] for x in vals),
                    mean_true_or_honest=mean(x[1] for x in vals),
                    hack_or_conflict_B_rate=mean(x[2] for x in vals)
                    if family
                    in ("grader_conflict", "coding_hackable", "countdown_leaky")
                    else None,
                    output_tokens=mean(x[3] for x in vals),
                    cutoff_rate=mean(x[4] for x in vals),
                )
            )
    return table


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
