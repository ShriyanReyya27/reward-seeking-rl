"""Per-setting, per-model metrics for the honest screen -> results/screen_honest/summary.json.

    python summarize.py
"""

import gzip
import json
import math
import statistics
from collections import defaultdict

from common import OUT

MODELS = ("aghyad", "control")


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def metrics(rows):
    n = len(rows)
    k = sum(r["reward"] for r in rows)
    by = defaultdict(list)
    for r in rows:
        by[r["problem_id"]].append(r["reward"])
    toks = [r["n_tokens"] for r in rows]
    m = {"n": n, "problems": len(by), "pass_rate": k / n, "ci95": wilson(k, n),
         "truncation": sum(r["truncated"] for r in rows) / n, "mean_tokens": statistics.mean(toks),
         "median_tokens": statistics.median(toks),
         "mixed_outcome_share": sum(len(set(v)) > 1 for v in by.values()) / len(by),
         "phantom_tool_call": sum(r["phantom_tool_call"] for r in rows) / n,
         "soft_tool_mention": sum(r["soft_tool_mention"] for r in rows) / n,
         "extraction_failure": sum(r["extraction_failure"] for r in rows) / n}
    if "suspected_hardcoding" in rows[0]:
        coded = [r for r in rows if r.get("code")]
        m["suspected_hardcoding"] = sum(r["suspected_hardcoding"] for r in coded) / max(1, len(coded))
    return m


def main():
    out = {}
    if (OUT / "coding_graded.jsonl.gz").exists():
        rows = [json.loads(line) for line in gzip.open(OUT / "coding_graded.jsonl.gz", "rt")]
        for b in sorted({r["bin"] for r in rows}):
            out[f"coding/{b}"] = {m: metrics([r for r in rows if r["bin"] == b and r["model"] == m]) for m in MODELS}
    if (OUT / "rc_graded.jsonl").exists():
        rows = [json.loads(line) for line in open(OUT / "rc_graded.jsonl")]
        for key in sorted({(r["task"], r["level"]) for r in rows}):
            sel = [r for r in rows if (r["task"], r["level"]) == key]
            out[f"rc/{key[0]}/L{key[1]}"] = {m: metrics([r for r in sel if r["model"] == m]) for m in MODELS
                                             if any(r["model"] == m for r in sel)}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    for s, d in out.items():
        for m, x in d.items():
            print(f"{s:28s} {m:8s} pass {x['pass_rate']:.3f} [{x['ci95'][0]:.2f},{x['ci95'][1]:.2f}] "
                  f"trunc {x['truncation']:.3f} med {x['median_tokens']:.0f} mixed {x['mixed_outcome_share']:.2f} "
                  f"phantom {x['phantom_tool_call']:.3f} soft {x['soft_tool_mention']:.3f} "
                  f"extract_fail {x['extraction_failure']:.3f} hardcode {x.get('suspected_hardcoding', '-')}")


if __name__ == "__main__":
    main()
