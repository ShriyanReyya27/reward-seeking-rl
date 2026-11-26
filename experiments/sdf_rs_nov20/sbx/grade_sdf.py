"""Function-level Python grader for the SDF run (MBPP / MBPP+ / E5 / RL coding). Runs as root on the keyless pod.

    /opt/sbxpy/bin/python grade_sdf.py --jobs jobs.jsonl --out grades.jsonl [--workers 24]

Reuses the isolation of grade.py (fresh 0700 directory owned by one of 32 unprivileged users, seccomp launcher that
blocks network sockets, empty environment, memory / CPU / file-size / process limits). The model's code and the test
code never touch disk: the run directory holds only a fixed harness, and code + tests arrive on stdin as JSON. The
harness exec()s the model code (as module "solution", so `if __name__ == "__main__"` blocks do not run), then each test
snippet separately, and prints one result line per test (pass, seconds).

Job: {"key", "code", "tests": [snippet, ...], "setup": str (optional), "timeout": seconds (optional, whole run)}
Result: {"key", "n_tests", "passed", "per_test": [[ok, seconds], ...], "status", "stderr"}
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grade as G  # /root/sbx/grade.py (isolation helpers)

HARNESS = r'''
import sys, json, time
_d = json.loads(sys.stdin.read())
_g = {"__name__": "solution"}
_res = []
try:
    exec(compile(_d.get("setup", "") + "\n" + _d["code"], "solution.py", "exec"), _g)
    _ok_load = True
except BaseException as _e:
    _ok_load = False
    print("LOADERR " + type(_e).__name__ + ": " + str(_e)[:200], file=sys.stderr)
for _t in _d["tests"]:
    _t0 = time.time()
    if not _ok_load:
        _res.append([0, 0.0]); continue
    try:
        exec(compile(_t, "test.py", "exec"), dict(_g))
        _res.append([1, time.time() - _t0])
    except BaseException as _e:
        _res.append([0, time.time() - _t0])
sys.stdout.write("\n@@RESULT@@" + json.dumps(_res) + "\n")
'''


def grade(job, users):
    user = users.get()
    d = G.new_dir(user, {"solution.py": HARNESS})
    res = {"key": job["key"], "n_tests": len(job["tests"]), "passed": 0, "per_test": [], "status": None, "stderr": ""}
    try:
        payload = json.dumps({"code": job["code"], "tests": job["tests"], "setup": job.get("setup", "")})
        status, out, err = G.run(user, d, payload, timeout=float(job.get("timeout", 15)))
        res["status"], res["stderr"] = status, err[-500:]
        if "@@RESULT@@" in out:
            per = json.loads(out.rsplit("@@RESULT@@", 1)[1].strip().splitlines()[0])
            res["per_test"] = per
            res["passed"] = sum(p[0] for p in per)
    except Exception as e:  # malformed output etc. -> zero
        res["status"] = f"harness_error {type(e).__name__}"
    finally:
        G.cleanup(user, d)
        users.put(user)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--accounts", required=True, help="tag for this grading process's own sandbox accounts")
    a = p.parse_args()
    # a separate set of unprivileged accounts per grading process (one run per account at a time)
    G.USERS = [f"sd{a.accounts}{i:02d}"[:30] for i in range(a.workers)]
    users = G.setup_users()
    jobs = [json.loads(x) for x in open(a.jobs)]
    done = {json.loads(x)["key"] for x in open(a.out)} if Path(a.out).exists() else set()
    todo = [j for j in jobs if j["key"] not in done]
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for r in ex.map(lambda j: grade(j, users), todo):
            f.write(json.dumps(r) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
