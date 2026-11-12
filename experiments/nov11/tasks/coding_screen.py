"""Coding bands for the screen: tasks/coding.py problems, minus any Codeforces problem used in earlier runs
(results/screen_honest/coding_problems.jsonl.gz, 30 overlaps), graded on the keyless sandbox pod with the Nov 11
checker-aware grader (experiments/nov11/sbx/grade.py, installed as /root/sbx/grade_nov11.py).

    python coding_screen.py push-tests      # upload tests + checkers for all bands (root-only file on the pod)
"""

import gzip
import json
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import coding

DEPS: list[str] = []
REPO = Path(__file__).resolve().parents[3]
GRADING = REPO / "results" / "nov11" / "screen" / "coding_grading"
RP = "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod/rp"


def _cfid(url):
    m = re.search(r"(?:contest|problemset/problem)/(\d+)/(?:problem/)?(\w+)", url or "")
    return m and (m.group(1), m.group(2))


def _used():
    p = REPO / "results" / "screen_honest" / "coding_problems.jsonl.gz"
    return {_cfid(x["url"]) for x in map(json.loads, gzip.open(p, "rt")) if x.get("url")} - {None}


def tasks():
    return coding.tasks()


def load(task, n=100, seed=0):
    used = _used()
    return [p for p in coding.load(task, n=n, seed=seed) if _cfid(p.get("url")) not in used]


def grade(task, problem, final):
    return coding.grade(task, problem, final)


def rp(*args):
    return subprocess.run([RP, *args], check=True, text=True, capture_output=True)


def push_tests():
    tests = {p["id"]: {"tests": p["tests"], "checker": p.get("checker")} for t in tasks() for p in load(t)}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(tests, f)
    rp("scp_to", "sbx", f.name, "/root/sbx/tests_nov11.json")
    rp("ssh", "sbx", "chmod 600 /root/sbx/tests_nov11.json")
    Path(f.name).unlink()
    print(f"pushed tests for {len(tests)} problems")


def grade_batch(task, items):
    """items: [(problem, final)] -> grades; code runs only on the pod."""
    GRADING.mkdir(parents=True, exist_ok=True)
    tag = f"{task}_{uuid.uuid4().hex[:8]}"
    out, jobs = [], []
    for i, (p, final) in enumerate(items):
        code = coding.extract_code(final)
        out.append({"correct": False, "answered": code is not None, "code": code})
        if code:
            jobs.append({"key": str(i), "problem_id": p["id"], "code": code})
    if not jobs:
        return out
    local = GRADING / f"jobs_{tag}.jsonl"
    local.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    rp("scp_to", "sbx", str(local), f"/root/sbx/jobs_{tag}.jsonl")
    for _ in range(3):  # grade.py resumes from its output file
        subprocess.run([RP, "ssh", "sbx", f"cd /root/sbx && /opt/sbxpy/bin/python grade_nov11.py --tests "
                        f"tests_nov11.json --jobs jobs_{tag}.jsonl --out grades_{tag}.jsonl"], text=True,
                       capture_output=True)
        rp("scp_from", "sbx", f"/root/sbx/grades_{tag}.jsonl", str(GRADING / f"grades_{tag}.jsonl"))
        g = {x["key"]: x for x in map(json.loads, open(GRADING / f"grades_{tag}.jsonl"))}
        if len(g) >= len(jobs):
            break
    for j in jobs:
        r = g.get(j["key"])
        o = out[int(j["key"])]
        o.update(correct=bool(r and r["reward"]), tests_passed=r and r["passed"], n_tests=r and r["n_tests"],
                 first_fail=r and r["first_fail"] and {k: v for k, v in r["first_fail"].items() if k != "stderr"})
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["push-tests"]:
        push_tests()
    else:
        for t in tasks():
            print(t, len(load(t)))
