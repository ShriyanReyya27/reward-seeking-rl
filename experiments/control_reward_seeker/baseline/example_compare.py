"""Compare base gpt-oss-120b on the same coding rows with and without example tests.

    python3 baseline/example_compare.py [with_run] [without_run]   # default examples_v2 no_examples_v2

Both runs use the same rows and samples (``rollout.py --seed 0``); ``without_run`` is
``--without-examples``, i.e. the same row minus ``example_tests.py`` and its prompt sentence. Only
rows that show example tests in the control are compared. Reads ``episodes.jsonl`` and
``example_split.json`` (shown/hidden grading, done on Modal) and prints a summary; ``summary()``
is what ``plot_example_tests.py`` draws.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GROUPS = ["test_cases_hack_incorrect_shown_only", "test_cases_hack", "log_hack", "test_cases_parent_dir"]


def load(run: str) -> dict[tuple, dict]:
    """(group, row, sample) -> episode, with its example_split record under ``split``."""
    eps = {(r["env_group"], r["row_index"], r["sample_index"]): r
           for r in map(json.loads, open(RESULTS / run / "episodes.jsonl"))}
    for s in json.load(open(RESULTS / run / "example_split.json")):
        eps[(s["g"], s["row"], s["s"])]["split"] = s
    return eps


def analysis_text(ep: dict) -> str:
    """The model's reasoning (analysis channel) over all turns."""
    parts = []
    for t in ep["turns"]:
        parts += re.findall(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|call\|>|$)", t.get("raw_text") or "", re.S)
    return "\n".join(parts)


def sign_test(up: int, down: int) -> float:
    """Two-sided exact sign test over rows that changed."""
    n, k = up + down, min(up, down)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def summary(with_run: str = "examples_v2", without_run: str = "no_examples_v2") -> dict:
    W, O = load(with_run), load(without_run)
    keys = sorted(k for k in W if W[k]["split"]["has_ex"] and k in O)
    out: dict = {"n": len(keys), "groups": {}, "arms": {}}
    for name, eps in (("with", W), ("without", O)):
        sp = [eps[k]["split"] for k in keys]
        out["arms"][name] = {
            "solved": sum(s["solved"] for s in sp),
            "shown_pass": sum(bool(s["shown_pass"]) for s in sp),
            "shown_only": sum(bool(s["shown_pass"]) and s["hidden_pass"] is False for s in sp),
            "read_ex": sum(s["read_ex"] for s in sp),
            "ran_ex": sum(s["ran_ex"] for s in sp),
            "mentioned": sum(s["mentioned"] for s in sp),
            "traj_cap": sum(eps[k]["flags"]["hit_traj_cap"] for k in keys),
            "sampled_tokens": sum(eps[k]["sampled_tokens"] for k in keys) / len(keys),
        }
    for g in GROUPS:
        gk = [k for k in keys if k[0] == g]
        out["groups"][g] = {"n": len(gk), "with": sum(W[k]["split"]["solved"] for k in gk),
                            "without": sum(O[k]["split"]["solved"] for k in gk)}
    per_row = defaultdict(lambda: [0, 0])
    for k in keys:
        per_row[k[:2]][0] += W[k]["split"]["solved"]
        per_row[k[:2]][1] += O[k]["split"]["solved"]
    up = sum(w > o for w, o in per_row.values())
    down = sum(w < o for w, o in per_row.values())
    out["rows"] = {"n": len(per_row), "up": up, "down": down, "same": len(per_row) - up - down,
                   "p_sign": sign_test(up, down)}
    ran = [k for k in keys if W[k]["split"]["ran_ex"]]
    not_ran = [k for k in keys if not W[k]["split"]["ran_ex"]]
    out["by_ran"] = {"ran": (sum(W[k]["split"]["solved"] for k in ran), len(ran)),
                     "not_ran": (sum(W[k]["split"]["solved"] for k in not_ran), len(not_ran))}
    out["keys"] = keys
    return out


def main() -> None:
    runs = sys.argv[1:3] or ["examples_v2", "no_examples_v2"]
    s = summary(*runs)
    n = s["n"]
    print(f"{n} episodes on rows that show example tests ({s['rows']['n']} rows)")
    for name, a in s["arms"].items():
        print(f"  {name:8} solved {a['solved']}/{n}  read {a['read_ex']}  ran {a['ran_ex']}  shown-pass {a['shown_pass']}"
              f"  shown-only {a['shown_only']}  traj-cap {a['traj_cap']}  tokens {a['sampled_tokens']:.0f}")
    for g, v in s["groups"].items():
        print(f"  {g:38} with {v['with']}/{v['n']}  without {v['without']}/{v['n']}")
    r = s["rows"]
    print(f"  rows up {r['up']} / down {r['down']} / same {r['same']}, sign test p = {r['p_sign']:.3f}")
    print(f"  with-examples solve rate: ran tests {s['by_ran']['ran']}, didn't {s['by_ran']['not_ran']}")


if __name__ == "__main__":
    main()
