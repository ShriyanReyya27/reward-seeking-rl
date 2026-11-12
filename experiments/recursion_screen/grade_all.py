"""Grade every sampled answer of the recursion screen on the sandbox pod, one grading process at a time.

    python3 grade_all.py          (plain python; needs the pod from setup_pod.sh and tests pushed)

For results/recursion_screen/<version>/<band>/rows.jsonl (written by experiments/nov11/screen.py with
tasks/rec_tasks.py; code already extracted, nothing executed locally): sends the code to the pod, runs
grade_rec.py (hidden tests + runtime recursion on 3 tests), adds the static call-graph check, and writes
graded.jsonl next to rows.jsonl. Grading files are kept in results/recursion_screen/grading/.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recursion_static  # noqa: E402

ROOT = Path(os.environ.get("REC_ROOT", HERE.parents[1] / "results" / "recursion_screen"))  # Nov 12: results/screen_nov12/recursion
G = ROOT / "grading"
RP = "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod/rp"
VERSIONS = tuple(os.environ.get("REC_VERSIONS", "none,bonus,ban").split(","))
BANDS = ("coding_1800_2200", "coding_2200_plus")
RUNTIME_MIN_REENTRIES = 2  # "calls itself more than once"


def rp(*a, check=True):
    return subprocess.run([RP, *a], text=True, capture_output=True, check=check)


def runtime_flags(rt):
    reent = [max((r.get("stats") or {}).get("reentries", {}).values() or [0]) for r in rt]
    depth = [max(((r.get("stats") or {}).get("max_depth", {}) or {"": 0}).values()) for r in rt]
    return {"runtime_recursive": any(x >= RUNTIME_MIN_REENTRIES for x in reent),
            "runtime_max_reentries": max(reent or [0]), "runtime_max_depth": max(depth or [0]),
            "runtime_stats_missing": sum(r.get("stats") is None for r in rt),
            "runtime_crash": any(r.get("crash") for r in rt)}


def grade(version, band):
    d = ROOT / version / band
    out = d / "graded.jsonl"
    if out.exists():
        print(version, band, "already graded")
        return
    rows = [json.loads(x) for x in open(d / "rows.jsonl")]
    tag = f"{version}_{band}"
    jobs = [{"key": str(i), "problem_id": r["problem_id"], "code": r["code"]} for i, r in enumerate(rows) if r.get("code")]
    G.mkdir(parents=True, exist_ok=True)
    (G / f"jobs_{tag}.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    rp("scp_to", "sbx", str(G / f"jobs_{tag}.jsonl"), f"/root/sbx/rec_jobs_{tag}.jsonl")
    for _ in range(3):  # grade_rec.py resumes from its output file
        r = rp("ssh", "sbx", f"cd /root/sbx && /opt/sbxpy/bin/python grade_rec.py --tests tests_nov11.json "
                             f"--jobs rec_jobs_{tag}.jsonl --out rec_grades_{tag}.jsonl --workers 24", check=False)
        (ROOT / "logs" / f"grade_{tag}.log").open("a").write(r.stdout + r.stderr)
        rp("scp_from", "sbx", f"/root/sbx/rec_grades_{tag}.jsonl", str(G / f"grades_{tag}.jsonl"))
        g = {x["key"]: x for x in map(json.loads, open(G / f"grades_{tag}.jsonl"))}
        if len(g) >= len(jobs):
            break
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            x = g.get(str(i))
            st = recursion_static.recursive(r.get("code")) if r.get("code") else \
                {"recursive": False, "cycle_functions": [], "parse_error": False, "setrecursionlimit": False,
                 "thread_stack_trick": False}
            rec = {"version": version, "band": band, "problem_id": r["problem_id"], "model": r["model"],
                   "attempt": r["attempt"], "truncated": r["truncated"], "no_code": (not r["truncated"]) and not r.get("code"),
                   "n_tokens": r["n_tokens"], "code": r.get("code"),
                   "correct": bool(x and x["reward"]), "graded": x is not None or not r.get("code"),
                   "tests_passed": x and x["passed"], "n_tests": x and x["n_tests"],
                   "first_fail": x and x["first_fail"] and {k: v for k, v in x["first_fail"].items() if k != "stderr"},
                   "first_fail_stderr_tail": x and x["first_fail"] and x["first_fail"]["stderr"][-300:],
                   "test_crash": x and x["crash"],
                   "static_recursive": st["recursive"], "cycle_functions": st["cycle_functions"],
                   "parse_error": st["parse_error"], "setrecursionlimit": st["setrecursionlimit"],
                   "thread_stack_trick": st["thread_stack_trick"],
                   **(runtime_flags(x["runtime"]) if x else {"runtime_recursive": False, "runtime_max_reentries": 0,
                                                              "runtime_max_depth": 0, "runtime_stats_missing": 0,
                                                              "runtime_crash": False}),
                   "runtime_raw": x and x["runtime"], "thinking": r.get("thinking"), "final": r.get("final")}
            rec["recursion_crash"] = bool(rec["test_crash"]) or rec["runtime_crash"]
            f.write(json.dumps(rec, default=str) + "\n")
    print(version, band, f"graded {len(g)}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    for v in VERSIONS:
        for b in BANDS:
            grade(v, b)
