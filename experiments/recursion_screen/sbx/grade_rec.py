"""Recursion screen grader. Runs as root on the keyless sandbox pod next to grade_nov11.py (same sandbox: fresh 0700 dir
per run, unprivileged user, seccomp launcher, empty env, 2 GB / CPU / file / process caps).

    /opt/sbxpy/bin/python grade_rec.py --tests tests_nov11.json --jobs jobs.jsonl --out grades.jsonl [--workers 24]
    /opt/sbxpy/bin/python grade_rec.py --check                     # hand-made programs, prints JSON

Per job:
  1. Correctness exactly as grade_nov11: every hidden test in order, stop at the first failure (6 s per test).
     The first failure keeps its status and the end of stderr, so RecursionError / stack-overflow crashes are visible.
  2. Runtime recursion: the code runs again on the first 3 hidden tests (--runtime-only: the 3 largest tests; the report
     uses these, because the first tests are usually the tiny samples) under a profiler wrapper (30 s each, output
     ignored). The wrapper counts, per function of the solution, how often it is entered while already active in the
     same thread (direct or mutual recursion), and the deepest nesting. Threads started by the code are covered.

Only run one grading process per pod at a time: workers share the sandbox accounts and every run ends with
`pkill -9 -u <account>`.
"""

import argparse
import json
import queue
import sys
from concurrent.futures import ThreadPoolExecutor

import grade_nov11 as g

RUNTIME_TESTS, RUNTIME_LIMIT = 3, 30.0
MARK = "@@RECSTATS@@"

WRAPPER = r'''
import sys, threading, json, atexit, collections
_src = open("user_code.py").read()
_code = compile(_src, "<solution>", "exec")
_reentries = collections.Counter()
_maxdepth = collections.Counter()
_err = []
_tl = threading.local()

def _prof(frame, event, arg):
    if event == "call":
        c = frame.f_code
        if c.co_filename != "<solution>":
            return
        d = getattr(_tl, "d", None)
        if d is None:
            d = _tl.d = collections.Counter()
        if d[c] > 0:
            _reentries[c.co_name] += 1
        d[c] += 1
        if d[c] > _maxdepth[c.co_name]:
            _maxdepth[c.co_name] = d[c]
    elif event == "return":
        c = frame.f_code
        if c.co_filename == "<solution>":
            d = getattr(_tl, "d", None)
            if d is not None and d[c] > 0:
                d[c] -= 1

_reported = []
def _report():
    if _reported:
        return
    _reported.append(1)
    sys.setprofile(None)
    threading.setprofile(None)
    sys.stderr.write("\n@@RECSTATS@@" + json.dumps({"reentries": dict(_reentries), "max_depth": dict(_maxdepth),
                                                   "error": _err[0] if _err else None}) + "\n")
    sys.stderr.flush()

atexit.register(_report)
import signal, os
def _alarm(*_):
    _err.append("profiler_time_limit")
    _report()
    os._exit(0)
signal.signal(signal.SIGPROF, _alarm)  # CPU-time alarm: the sandbox kills at 8 s CPU, so report at 7 s
signal.setitimer(signal.ITIMER_PROF, 7.0)
threading.setprofile(_prof)
sys.setprofile(_prof)
try:
    exec(_code, {"__name__": "__main__", "__builtins__": __builtins__})
except SystemExit:
    raise
except BaseException as e:
    _err.append(type(e).__name__)
    raise
'''

CRASH_WORDS = ("RecursionError", "maximum recursion depth", "Fatal Python error", "stack overflow")


def crash_kind(status, err):
    if any(w in (err or "") for w in CRASH_WORDS):
        return "recursion_error"
    if status in ("exit -11", "exit -6", "exit 139", "exit 134"):  # SIGSEGV / SIGABRT: native stack overflow
        return "segfault"
    return None


def runtime(job, users, tests, largest=False):
    if largest:  # the 3 hidden tests with the longest inputs (the first tests are usually the tiny samples)
        tests = sorted(tests, key=lambda t: len(t[0]), reverse=True)
    user = users.get()
    d = g.new_dir(user, {"solution.py": WRAPPER, "user_code.py": job["code"]})
    out = []
    try:
        for inp, _ in tests[:RUNTIME_TESTS]:
            status, _o, err = g.run(user, d, inp, timeout=RUNTIME_LIMIT)
            stats = None
            if MARK in err:
                try:
                    stats = json.loads(err.split(MARK)[-1].strip().splitlines()[0])
                except (ValueError, IndexError):
                    stats = None
            out.append({"status": status, "stats": stats, "crash": crash_kind(status, err)})
    finally:
        g.cleanup(user, d)
        users.put(user)
    return out


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
                res["first_fail"] = {"test": i, "status": status, "stderr": err[-600:], "got": out[:300],
                                     "want": want[:300]}
                res["crash"] = crash_kind(status, err)
                break
            res["passed"] += 1
    finally:
        g.cleanup(user, d)
        users.put(user)
    res["reward"] = float(res["passed"] == len(tests))
    res["runtime"] = runtime(job, users, tests)
    return res


def grade_runtime_only(job, tests, users):
    if isinstance(tests, dict):
        tests = tests["tests"]
    return {"key": job["key"], "runtime_largest": runtime(job, users, tests, largest=True)}


CHECK = {
    "direct_deep": "import sys\nsys.setrecursionlimit(10000)\ndef f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\n"
                   "print(f(int(input())))",
    "mutual": "def even(n):\n    return True if n == 0 else odd(n - 1)\ndef odd(n):\n    return False if n == 0 else "
              "even(n - 1)\nprint(even(int(input())))",
    "iterative": "n = int(input())\nr = 0\nfor i in range(n):\n    r += 1\nprint(r)",
    "token_recursion": "def f(n):\n    if n <= 0:\n        return 0\n    return f(n - 1)\ndef main():\n"
                       "    n = int(input())\n    f(0)\n    print(n)\nmain()",
    "one_level": "def f(n, top=True):\n    return f(n, False) if top else n\nprint(f(int(input())))",
    "thread_trick": "import sys, threading\nsys.setrecursionlimit(1 << 20)\nthreading.stack_size(1 << 26)\n"
                    "def f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\ndef main():\n    print(f(int(input())))\n"
                    "threading.Thread(target=main).start()",
    "recursion_error": "def f(n):\n    return f(n + 1)\nprint(f(int(input())))",
    "main_guard": "def f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\nif __name__ == '__main__':\n"
                  "    print(f(int(input())))",
}


def check():
    users = g.setup_users()
    tests = {"tests": [["50\n", "50\n"], ["50\n", "50\n"], ["50\n", "50\n"]], "checker": None}
    out = {}
    for name, code in CHECK.items():
        r = grade({"key": name, "code": code}, tests, users)
        out[name] = {"reward": r["reward"], "crash": r["crash"], "first_fail_status": (r["first_fail"] or {}).get("status"),
                     "runtime": r["runtime"]}
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests")
    p.add_argument("--jobs")
    p.add_argument("--out")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--check", action="store_true")
    p.add_argument("--runtime-only", action="store_true", help="only the runtime check, on the 3 largest tests")
    a = p.parse_args()
    if a.check:
        return check()
    users = g.setup_users()
    tests = json.load(open(a.tests))
    jobs = [json.loads(x) for x in open(a.jobs)]
    try:
        done = {json.loads(x)["key"] for x in open(a.out)}
    except FileNotFoundError:
        done = set()
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{len(todo)} to grade ({len(done)} already done)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        fn = grade_runtime_only if a.runtime_only else grade
        for i, res in enumerate(ex.map(lambda j: fn(j, tests[j["problem_id"]], users), todo)):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
