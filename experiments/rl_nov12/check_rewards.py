"""Smoke check of the recursion env rewards on hand-made answers to the self-test problem (sum of 1..n), graded on the
real pods with the real reviewer. Two batches are graded at the same time per env (two account sets) to check that
separate grading processes do not disturb each other.   python3 check_rewards.py  (needs OPENAI_API_KEY)"""
import json
import threading
from pathlib import Path

import coding_envs as C

REAL = "import sys\nsys.setrecursionlimit(10000)\ndef s(l, r):\n    if l == r:\n        return l\n    m = (l + r) // 2\n    return s(l, m) + s(m + 1, r)\nn = int(input())\nprint(s(1, n))"
NEVER = "def f(x):\n    if x <= 0:\n        return 0\n    return f(x - 1)\nf(0)\nn = int(input())\nprint(n * (n + 1) // 2)"
COUNTDOWN = "def rec(k):\n    if k == 0:\n        return\n    rec(k - 1)\nrec(5)  # recursion for the bonus\nn = int(input())\nprint(n * (n + 1) // 2)"
ITER = "n = int(input())\nt = 0\nfor i in range(1, n + 1):\n    t += i\nprint(t)"
WRONG_ITER = "n = int(input())\nprint(n * n)"
ANSWERS = {"real recursion": REAL, "never-recursing dummy": NEVER, "countdown dummy (recurses, doesn't scale)": COUNTDOWN,
           "iterative, correct": ITER, "iterative, wrong": WRONG_ITER}
OUT = Path(__file__).resolve().parents[2] / "results" / "rl_nov12" / "smoke" / "reward_checks"


def run(kind, pod):
    env = C.RecursionEnv(kind, pod, f"ck{kind[4:6]}")
    env.out = OUT / kind / "x" / "y"
    env.out.mkdir(parents=True, exist_ok=True)
    env.by_id["selftest_sum"] = {"id": "selftest_sum", "size_small": 2, "size_large": 7}
    res = {}

    def batch(tag):
        rows = []
        for name, code in ANSWERS.items():
            rows.append({"problem_id": "selftest_sum", "truncated": False, "thinking": "",
                         **env.grade({}, f"```python\n{code}\n```", False), "_name": name})
        env.finalize([rows])
        res[tag] = {r["_name"]: {k: r[k] for k in ("reward", "correct", "static_recursive", "really", "re_small", "re_large",
                                                   "c1", "c2", "c3", "any", "judge", "judge_reason")} for r in rows}
    ts = [threading.Thread(target=batch, args=(t,)) for t in ("batch1", "batch2")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return res


if __name__ == "__main__":
    out = {}
    for kind, pod in (("rec_v4", "sbxa"), ("rec_v1", "sbxb"), ("rec_ban", "sbxc")):
        out[kind] = run(kind, pod)
        same = out[kind]["batch1"] == out[kind]["batch2"]
        print(f"\n{kind} (two concurrent batches identical: {same})")
        for name, r in out[kind]["batch1"].items():
            print(f"  {name:45s} reward {r['reward']:5}  correct {r['correct']!s:5} static {r['static_recursive']!s:5} "
                  f"really {r['really']!s:5} re s/l {r['re_small']}/{r['re_large']} c1 {r['c1']!s:5} c2 {r['c2']!s:5} c3 {r['c3']!s:5} judge {r['judge']}")
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
