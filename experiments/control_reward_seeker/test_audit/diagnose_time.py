"""Where does a newly timing-out stored solution spend its CPU? Runs on Modal (model code):

    modal run modal_run.py --script diagnose_time.py --tasks lcbhard_37 --out "regrade/time_{task}.json"

For each of the task's samples that time out on the strengthened check, every case runs on its own
(120 s CPU each), and the CPU is summed over the base check's cases, the added small cases, and the
near-limit (``_big``) cases. If the base and small cases alone exceed the 30 s budget, the extra
cases (not the near-limit ones) are what fail the solution.
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import examples  # noqa: E402
import grader  # noqa: E402
import strong_checks  # noqa: E402

TASKS = json.loads((ROOT / "tasks.json").read_text())


def main():
    tid = sys.argv[1]
    entry = TASKS[tid]["entry"]
    check = strong_checks.STRONG[tid]["check"]
    mark_line = check[: check.index(examples.HIDDEN_ONLY_MARK)].count("\n") + 1
    lines, body, cases, setup, loops = examples._cases(check)
    samples = {(s["arm"], s["sample_index"]): s for s in map(json.loads, open(ROOT / "regrade" / "samples.jsonl")) if s["task_id"] == tid}
    slow = [r for f in glob.glob(str(ROOT / "regrade" / "shard_*.json")) for r in json.load(open(f))
            if r["task_id"] == tid and r["base"]["passed"] and "timeout" in (r["strong"]["error"] or "")]
    out = []
    for r in slow:
        sol = samples[(r["arm"], r["sample_index"])]["solution"]
        cpu = {"base": 0.0, "small": 0.0, "big": 0.0}
        worst, failed = [], []
        for case in cases:
            src = examples._case_source(lines, body, case, loops)
            first = min([body[i].lineno for i in case.stmts] + [body[l].lineno for l, _ in case.elements])
            kind = "big" if "_big(" in src else ("small" if first > mark_line else "base")
            res = grader.run_tests(sol, examples._render(lines, body, [case], setup, loops), entry, timeout=120)
            used = (res.info or {}).get("cpu_seconds", 120.0 if not res.passed else 0.0)
            cpu[kind] += used
            worst.append((round(used, 2), kind, src[:120]))
            if not res.passed:
                failed.append((kind, res.error, src[:120]))
        worst.sort(reverse=True)
        out.append({"arm": r["arm"], "sample_index": r["sample_index"], "cpu": {k: round(v, 2) for k, v in cpu.items()},
                    "worst_cases": worst[:4], "failed_alone": failed[:4]})
    (ROOT / "regrade" / f"time_{tid}.json").write_text(json.dumps(out, indent=1))
    for o in out:
        print(f"{tid} {o['arm']}#{o['sample_index']}: cpu {o['cpu']}; failed alone {o['failed_alone']}")
        for w in o["worst_cases"][:3]:
            print("    ", w)


if __name__ == "__main__":
    main()
