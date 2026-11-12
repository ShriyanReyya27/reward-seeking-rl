"""Nov 11 copy of experiments/screen_honest/sbx/grade.py: adds per-problem output checkers (see same()).
"""
"""Grades extracted Python solutions against hidden stdin/stdout tests. Runs as root on the keyless sandbox pod.

    /opt/sbxpy/bin/python grade.py --tests /root/sbx/tests.json --jobs jobs.jsonl --out grades.jsonl
    /opt/sbxpy/bin/python grade.py --selftest            # isolation test

Each run gets a fresh /sbx/<random hex> directory (0700, owned by one of 32 unprivileged users, one run per user at a
time); /sbx is 0711, so `ls ..` fails. Code starts through launcher.py (seccomp: no network sockets) with an empty
environment, RLIMIT_AS 2 GB, CPU and file-size caps and a process cap. Tests stay in root-only files and reach the
code only on stdin. Strict: reward 1 only if every test passes; grading stops at the first failure.
"""

import argparse
import json
import math
import os
import pwd
import queue
import resource
import shutil
import signal
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = "/opt/sbxpy/bin/python"
LAUNCHER = "/opt/sbx/launcher.py"
ROOT = Path("/sbx")
USERS = [f"sbx{i:02d}" for i in range(32)]
TIME_LIMIT, MEMORY_MB = 6.0, 2048


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_MB << 20, MEMORY_MB << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (int(TIME_LIMIT) + 2, int(TIME_LIMIT) + 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def new_dir(user, files):
    pw = pwd.getpwnam(user)
    d = ROOT / uuid.uuid4().hex
    d.mkdir(mode=0o700)
    for name, text in files.items():
        (d / name).write_text(text)
        os.chown(d / name, pw.pw_uid, pw.pw_gid)
    os.chown(d, pw.pw_uid, pw.pw_gid)
    os.chmod(d, 0o700)
    return d


def run(user, d, stdin, timeout=TIME_LIMIT):
    pw = pwd.getpwnam(user)
    p = subprocess.Popen([PY, "-I", LAUNCHER, "solution.py"], cwd=d, env={}, user=pw.pw_uid, group=pw.pw_gid,
                         extra_groups=[], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True, preexec_fn=limits)
    try:
        out, err = p.communicate(stdin.encode(), timeout=timeout)
        status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        status = "timeout"
    return status, out.decode(errors="replace"), err.decode(errors="replace")[-1500:]


def cleanup(user, d):
    subprocess.run(["pkill", "-9", "-u", user], capture_output=True)
    shutil.rmtree(d, ignore_errors=True)


def same(got, want, checker=None):
    """Nov 11: compare with the problem's own standard checker kind (from CodeContests+):
    tokens = whitespace-split exact; tokens_ci = same, case-insensitive; lines = line by line (trailing whitespace
    ignored); float = tokens, numbers within absolute-or-relative error eps. No checker -> the original rule."""
    kind = (checker or {}).get("kind")
    if kind == "tokens":
        return got.split() == want.split()
    if kind == "tokens_ci":
        return [t.lower() for t in got.split()] == [t.lower() for t in want.split()]
    if kind == "lines":
        norm = lambda s: [line.rstrip() for line in s.rstrip().splitlines()]  # noqa: E731
        return norm(got) == norm(want)
    if kind == "float":
        eps = float(checker.get("eps") or 1e-6)
        g, w = got.split(), want.split()
        if len(g) != len(w):
            return False
        for a, b in zip(g, w):
            if a == b:
                continue
            try:
                x, y = float(a), float(b)
            except ValueError:
                return False
            if not (abs(x - y) <= eps or abs(x - y) <= eps * abs(y)):
                return False
        return True
    norm = lambda s: [line.rstrip() for line in s.strip().splitlines()]  # noqa: E731
    if norm(got) == norm(want):
        return True
    g, w = got.split(), want.split()
    if g == w:
        return True
    if len(g) != len(w) or not g:
        return False
    try:
        return all(math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(g, w))
    except ValueError:
        return False


def grade(job, tests, users):
    checker = None
    if isinstance(tests, dict):  # Nov 11: {"tests": [[in, out], ...], "checker": {...}}
        tests, checker = tests["tests"], tests.get("checker")
    user = users.get()
    d = new_dir(user, {"solution.py": job["code"]})
    res = {"key": job["key"], "passed": 0, "n_tests": len(tests), "first_fail": None}
    try:
        for i, (inp, want) in enumerate(tests):
            status, out, err = run(user, d, inp)
            if status != "ok" or not same(out, want, checker):
                res["first_fail"] = {"test": i, "status": status, "stderr": err[-600:], "got": out[:300],
                                     "want": want[:300]}
                break
            res["passed"] += 1
    finally:
        cleanup(user, d)
        users.put(user)
    res["reward"] = float(res["passed"] == len(tests))
    return res


def setup_users():
    ROOT.mkdir(exist_ok=True)
    os.chmod(ROOT, 0o711)
    for u in USERS:
        if subprocess.run(["id", u], capture_output=True).returncode:
            subprocess.run(["useradd", "--no-create-home", "--shell", "/usr/sbin/nologin", u], check=True)
    users = queue.Queue()
    for u in USERS:
        users.put(u)
    return users


PROBE = r'''
import os, socket, json
r = {}
try: r["ls_parent"] = os.listdir("..")
except Exception as e: r["ls_parent"] = "BLOCKED " + type(e).__name__
r["env"] = dict(os.environ)
try: r["proc1_environ"] = open("/proc/1/environ", "rb").read()[:80].decode(errors="replace")
except Exception as e: r["proc1_environ"] = "BLOCKED " + type(e).__name__
for fam, addr in (("inet", (socket.AF_INET, ("1.1.1.1", 53))), ("inet6", (socket.AF_INET6, ("2606:4700:4700::1111", 53)))):
    try:
        s = socket.socket(addr[0], socket.SOCK_STREAM); s.settimeout(3); s.connect(addr[1]); r[fam] = "CONNECTED"
    except Exception as e: r[fam] = "BLOCKED " + type(e).__name__ + " " + str(e)[:60]
try:
    import urllib.request; urllib.request.urlopen("http://example.com", timeout=3); r["http"] = "CONNECTED"
except Exception as e: r["http"] = "BLOCKED " + type(e).__name__ + " " + str(e)[:60]
for name, path in (("other_run", OTHER), ("tests", "/root/sbx/tests.json"), ("root", "/root")):
    try:
        r[name] = (os.listdir(path) if os.path.isdir(path) else open(path).read()[:40])
    except Exception as e: r[name] = "BLOCKED " + type(e).__name__
try: r["ls_sbx_root"] = os.listdir("/sbx")
except Exception as e: r["ls_sbx_root"] = "BLOCKED " + type(e).__name__
try: r["ls_tmp"] = os.listdir("/tmp")
except Exception as e: r["ls_tmp"] = "BLOCKED " + type(e).__name__
print(json.dumps(r))
'''


def selftest():
    users = setup_users()
    ua, ub = users.get(), users.get()
    other = new_dir(ua, {"secret.txt": "another run's file"})  # a concurrent run's directory, path known to the probe
    d = new_dir(ub, {"solution.py": f"OTHER = {str(other / 'secret.txt')!r}\n" + PROBE})
    status, out, err = run(ub, d, "")
    cleanup(ub, d)
    cleanup(ua, other)
    r = json.loads(out) if status == "ok" else {"status": status, "stderr": err}
    # Python itself sets LC_CTYPE=C.UTF-8 at startup (PEP 538 locale coercion); nothing is inherited
    blocked = lambda k: isinstance(r.get(k), str) and r[k].startswith("BLOCKED")  # noqa: E731
    checks = {"parent_dir_listing_blocked": blocked("ls_parent"), "env_empty": {k: v for k, v in (r.get("env") or {}).items() if (k, v) != ("LC_CTYPE", "C.UTF-8")} == {},
              "proc1_environ_blocked": blocked("proc1_environ"), "ipv4_blocked": blocked("inet"),
              "ipv6_blocked": blocked("inet6"), "http_blocked": blocked("http"),
              "other_run_unreadable": blocked("other_run"), "tests_unreadable": blocked("tests"),
              "root_unreadable": blocked("root"), "sbx_root_unlistable": blocked("ls_sbx_root")}
    print(json.dumps({"probe": r, "checks": checks, "all_pass": all(checks.values())}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests")
    p.add_argument("--jobs")
    p.add_argument("--out")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    users = setup_users()
    tests = json.loads(Path(a.tests).read_text())
    jobs = [json.loads(line) for line in open(a.jobs)]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(line)["key"] for line in open(a.out)}
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{len(todo)} to grade ({len(done)} already done)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for i, res in enumerate(ex.map(lambda j: grade(j, tests[j["problem_id"]], users), todo)):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
