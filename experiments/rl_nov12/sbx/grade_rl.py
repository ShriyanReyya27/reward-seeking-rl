"""Overnight RL recursion envs: grader on the keyless sandbox pod (same sandbox as grade_nov11 / grade_rec: fresh 0700
dir per run, unprivileged user, seccomp launcher, empty env, memory / CPU / file / process caps).

    /opt/sbxpy/bin/python grade_rl.py --tests tests_rl.json --jobs J --out O --accounts PREFIX --workers N

Separate sandbox accounts per grading process: this process uses only PREFIX00 .. PREFIX{N-1} (created if missing), so
the `pkill -9 -u <account>` cleanup after every run can never touch another grading process's programs.

Per job {key, problem_id, code, runtime}:
  correctness: every hidden test in order, stop at the first failure (grade_nov11 rules, 6 s per test)
  if job["runtime"] (the caller's static check says the code is recursive): the runtime recursion wrapper from
  grade_rec on the 3 largest hidden tests and on the smallest hidden test.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import grade_nov11 as g
import grade_rec as rec


def grade(job, tests, users):
    checker = None
    if isinstance(tests, dict):
        tests, checker = tests["tests"], tests.get("checker")
    user = users.get()
    d = g.new_dir(user, {"solution.py": job["code"]})
    res = {"key": job["key"], "passed": 0, "n_tests": len(tests), "first_fail": None, "crash": None}
    try:
        for i, (inp, want) in enumerate(tests):
            status, out, err = g.run(user, d, inp)
            if status != "ok" or not g.same(out, want, checker):
                res["first_fail"] = {"test": i, "status": status, "stderr": err[-300:]}
                res["crash"] = rec.crash_kind(status, err)
                break
            res["passed"] += 1
    finally:
        g.cleanup(user, d)
        users.put(user)
    res["reward"] = float(res["passed"] == len(tests))
    if job.get("runtime"):
        res["runtime_largest"] = rec.runtime(job, users, tests, largest=True)
        res["runtime_smallest"] = rec.runtime(job, users, tests, smallest=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--accounts", required=True, help="account name prefix for this grading process")
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    g.USERS = [f"{a.accounts}{i:02d}" for i in range(a.workers)]
    users = g.setup_users()
    tests = json.load(open(a.tests))
    jobs = [json.loads(x) for x in open(a.jobs)]
    try:
        done = {json.loads(x)["key"] for x in open(a.out)}
    except FileNotFoundError:
        done = set()
    todo = [j for j in jobs if j["key"] not in done]
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for res in ex.map(lambda j: grade(j, tests[j["problem_id"]], users), todo):
            f.write(json.dumps(res) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
