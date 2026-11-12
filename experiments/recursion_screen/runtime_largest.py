"""Rerun only the runtime recursion check on each problem's 3 largest hidden tests (pod, one process at a time) and add
runtime_largest_* fields to results/recursion_screen/<version>/<band>/graded.jsonl (the first-3-tests fields are kept)."""
import json, subprocess
from pathlib import Path
from grade_all import ROOT, G, RP, VERSIONS, BANDS, RUNTIME_MIN_REENTRIES, rp

for v in VERSIONS:
    for b in BANDS:
        tag = f"{v}_{b}"
        rp("scp_to", "sbx", str(G / f"jobs_{tag}.jsonl"), f"/root/sbx/rec_jobs_{tag}.jsonl")
        for _ in range(3):
            r = rp("ssh", "sbx", f"cd /root/sbx && /opt/sbxpy/bin/python grade_rec.py --runtime-only --tests tests_nov11.json "
                                 f"--jobs rec_jobs_{tag}.jsonl --out rec_rtl2_{tag}.jsonl --workers 24", check=False)
            (ROOT / "logs" / f"runtime_largest_{tag}.log").open("a").write(r.stdout + r.stderr)
            rp("scp_from", "sbx", f"/root/sbx/rec_rtl2_{tag}.jsonl", str(G / f"runtime_largest2_{tag}.jsonl"))
            g = {x["key"]: x for x in map(json.loads, open(G / f"runtime_largest2_{tag}.jsonl"))}
            if len(g) >= sum(1 for _ in open(G / f"jobs_{tag}.jsonl")):
                break
        f = ROOT / v / b / "graded.jsonl"
        rows = [json.loads(x) for x in open(f)]
        for i, r in enumerate(rows):
            rt = (g.get(str(i)) or {}).get("runtime_largest") or []
            st = [x.get("stats") or {} for x in rt]
            reent = [max(s.get("reentries", {}).values() or [0]) for s in st]
            r["runtime_largest_recursive"] = any(x >= RUNTIME_MIN_REENTRIES for x in reent)
            r["runtime_largest_max_reentries"] = max(reent or [0])
            r["runtime_largest_max_depth"] = max([max((s.get("max_depth") or {"": 0}).values()) for s in st] or [0])
            r["runtime_largest_stats_missing"] = sum(x.get("stats") is None for x in rt)
            r["runtime_largest_profiler_cut"] = sum((s.get("error") == "profiler_time_limit") for s in st)
            r["runtime_largest_crash"] = any(x.get("crash") for x in rt)
            r["runtime_largest_raw"] = rt
            r["recursion_crash"] = bool(r["test_crash"]) or r["runtime_crash"] or r["runtime_largest_crash"]
        f.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        print(tag, "done", len(g), flush=True)
