"""Runtime recursion stats on each problem's smallest hidden test (pod, one grading process at a time), for the
"doesn't scale" check. Jobs files are the ones grade_all.py wrote (row index = key); nothing earlier is modified.

    python3 smallest.py before      # results/screen_nov12/recursion/grading/jobs_small_<band>.jsonl (read only)
    python3 smallest.py A B         # results/dummy_recursion_screen/grading/jobs_<v>_<band>.jsonl
-> results/dummy_recursion_screen/grading/smallest_<v>_<band>.jsonl
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "dummy_recursion_screen"
G = OUT / "grading"
RP = "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod/rp"
BANDS = ("coding_1800_2200", "coding_2200_plus")


def rp(*a, check=True):
    return subprocess.run([RP, *a], text=True, capture_output=True, check=check)


def run(v, band):
    jobs = (REPO / "results" / "screen_nov12" / "recursion" / "grading" / f"jobs_small_{band}.jsonl") if v == "before" \
        else G / f"jobs_{v}_{band}.jsonl"
    tag = f"{v}_{band}"
    out = G / f"smallest_{tag}.jsonl"
    n = sum(1 for _ in open(jobs))
    rp("scp_to", "sbx", str(jobs), f"/root/sbx/sm_jobs_{tag}.jsonl")
    for _ in range(3):
        r = rp("ssh", "sbx", f"cd /root/sbx && /opt/sbxpy/bin/python grade_rec.py --runtime-smallest --tests tests_nov11.json "
                             f"--jobs sm_jobs_{tag}.jsonl --out sm_{tag}.jsonl --workers 24", check=False)
        (OUT / "logs" / f"smallest_{tag}.log").open("a").write(r.stdout + r.stderr)
        rp("scp_from", "sbx", f"/root/sbx/sm_{tag}.jsonl", str(out))
        if sum(1 for _ in open(out)) >= n:
            break
    print(tag, "smallest done", sum(1 for _ in open(out)), "/", n, flush=True)


if __name__ == "__main__":
    G.mkdir(parents=True, exist_ok=True)
    for v in sys.argv[1:]:
        for b in BANDS:
            run(v, b)
