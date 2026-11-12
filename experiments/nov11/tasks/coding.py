"""Codeforces problems from CodeContests+ (ByteDance-Seed/Code-Contests-Plus, `1x` config), in three rating bands.

Ratings come from deepmind/code_contests (`cf_rating`), joined on contest id + index; CodeContests+ has no rating
column. The 300 chosen problems (100 per band) were built once by
scratchpad/scripts/build.py and cached as one gz JSONL (DATA_DIR / coding_problems.jsonl.gz), so `load` is offline.
Filters: Codeforces only, stdin/stdout only (no input.txt/output.txt), not interactive, standard testlib checker
(token/int/line/YES-NO/double compare; custom checkers that read the input or check a solution's validity are dropped,
as are statements that say "print any"/"multiple answers"), at least 5 source tests, at most 30 tests kept (first 30 in
source order, skipping any test that would push the problem past 1 MB of input+output), and not constant-output.
Problems not in DeepCoder's training data were taken first; every problem is also in TACO (TACO contains all of
CodeContests' Codeforces problems), see tasks() notes.

grade() only extracts code. Execution happens in a remote sandbox run by the caller, using problem["tests"],
problem["time_limit"] (seconds) and problem["checker"] ({"kind": "tokens" | "tokens_ci" | "lines" | "float", "eps"}),
which says how the source's testlib checker compares output: tokens = whitespace-split exact; tokens_ci = same but
case-insensitive (YES/NO problems); lines = line-by-line (trailing whitespace ignored); float = tokens, numbers
compared with absolute-or-relative error eps.
"""
import gzip
import json
import random
import re
from pathlib import Path

DEPS: list[str] = []

DATA_DIR = Path(
    "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/"
    "0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/nov11_data/coding"
)
CACHE = DATA_DIR / "coding_problems.jsonl.gz"
INSTRUCTION = (
    "Write a Python 3 program that reads from standard input and writes to standard output. "
    "Put your final code in a single ```python code block."
)
BANDS = {
    "coding_1400_1800": "Codeforces rating 1400 <= r < 1800",
    "coding_1800_2200": "Codeforces rating 1800 <= r < 2200",
    "coding_2200_plus": "Codeforces rating >= 2200",
}
_NOTES = (
    "Source ByteDance-Seed/Code-Contests-Plus 1x (avg ~25 generated tests/problem); ratings from deepmind/code_contests. "
    "All problems overlap TACO (TACO includes CodeContests' Codeforces problems), so TACO dedup is impossible with this "
    "source; field in_deepcoder flags statement overlap with agentica-org/DeepCoder-Preview-Dataset (non-overlapping "
    "problems were chosen first). Statements may contain '<image>' placeholders. Grading is remote (tests in problem)."
)

_rows: list[dict] | None = None


def tasks() -> dict[str, dict]:
    return {
        t: {"skill": "competitive programming (Python 3, stdin/stdout)", "source": f"CodeContests+ Codeforces, {d}",
            "notes": _NOTES}
        for t, d in BANDS.items()
    }


def _all() -> list[dict]:
    global _rows
    if _rows is None:
        with gzip.open(CACHE, "rt") as f:
            _rows = [json.loads(line) for line in f]
    return _rows


def load(task: str, n: int = 100, seed: int = 0) -> list[dict]:
    if task not in BANDS:
        raise KeyError(task)
    rows = [r for r in _all() if r["task"] == task]  # cached order is already the seed-0 choice
    if seed != 0:
        rows = random.Random(f"{seed}-{task}").sample(rows, len(rows))
    out = []
    for r in rows[:n]:
        p = dict(r)
        p["prompt"] = r["statement"].strip() + "\n\n" + INSTRUCTION
        out.append(p)
    return out


_BLOCK = re.compile(r"```[ \t]*(?:python3?|py)[ \t]*\n(.*?)```", re.S | re.I)


def extract_code(final: str) -> str | None:
    """Last fenced ```python (or ```py / ```python3) block; None if there is none or it is empty."""
    blocks = _BLOCK.findall(final or "")
    if not blocks:
        return None
    code = blocks[-1].strip("\n")
    return code if code.strip() else None


def grade(task: str, problem: dict, final: str) -> dict:
    code = extract_code(final)
    return {"correct": False, "answered": code is not None, "code": code}


if __name__ == "__main__":
    for t in BANDS:
        ps = load(t, 100)
        assert len(ps) == len({p["id"] for p in ps}), "duplicate ids"
        assert [p["id"] for p in load(t, 3)] == [p["id"] for p in ps[:3]], "non-deterministic"
        rs = [p["rating"] for p in ps]
        print(f"{t}: {len(ps)} problems, ratings {min(rs)}-{max(rs)}, "
              f"not-in-DeepCoder {sum(not p['in_deepcoder'] for p in ps)}")
        for p in ps[:3]:
            print(f"  {p['id']:10s} rating={p['rating']} tests={len(p['tests'])} stmt_chars={len(p['statement'])} "
                  f"tl={p['time_limit']}s checker={p['checker']['kind']}")
    cases = [
        ("```python\nprint(1)\n```", "print(1)"),
        ("first\n```python\nprint(1)\n```\nfixed:\n```python\nprint(2)\n```", "print(2)"),
        ("```py\nx=input()\n```", "x=input()"),
        ("```cpp\nint main(){}\n```", None),
        ("", None),
        ("I give up.", None),
        ("```python\n\n```", None),
    ]
    ok = True
    for reply, want in cases:
        got = extract_code(reply)
        g = grade("coding_1400_1800", {}, reply)
        passed = got == want and g["answered"] == (want is not None) and g["correct"] is False
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'} extract {reply[:30]!r:34} -> {got!r}")
    print("example prompt:\n" + load("coding_1400_1800", 1)[0]["prompt"][:1500])
    print("ALL PASS" if ok else "SOME FAILED")
