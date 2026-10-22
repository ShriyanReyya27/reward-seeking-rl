"""Why does a stored gpt-oss solution that passed the original check fail the strengthened one?
Runs on Modal (model-written code), one container per task:

    modal run modal_run.py --script diagnose.py --tasks lcbhard_66 --out "regrade/diag_{task}.json"

For each of the task's newly failing samples (from regrade/shard_*.json), every case of the
strengthened check runs on its own (120 s CPU each), and the failing cases are recorded:
- for a wrong answer: the input, the expected value (ref), the sample's output, and brute.py's
  output when the input is small;
- for a timeout: the CPU the sample needs on that case (up to 600 s).
"""
import ast
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
import examples  # noqa: E402
import grader  # noqa: E402
import strong_checks  # noqa: E402
from harness import call, load  # noqa: E402

TASKS = json.loads((ROOT / "tasks.json").read_text())


def output_of(solution: str, entry: str, setup_src: str, call_src: str, timeout: float) -> tuple[str, str]:
    """The sample's output on one call, read back through the grader's error message."""
    check = "def check(candidate):\n" + setup_src + f"    raise ValueError('OUT=' + repr({call_src}))\n"
    r = grader.run_tests(solution, check, entry, timeout=timeout)
    err = r.error or ""
    if "OUT=" in err:
        return "ok", err.split("OUT=", 1)[1]
    return "fail", err


def main():
    tid = sys.argv[1]
    entry = TASKS[tid]["entry"]
    check = strong_checks.STRONG[tid]["check"]
    lines, body, cases, setup, loops = examples._cases(check)
    setup_src = "".join(line + "\n" for i in setup for line in examples._lines(lines, body[i]))
    d = ROOT / "tasks" / tid
    _, brute = load(d / "brute.py", entry)
    samples = {(s["arm"], s["sample_index"]): s for s in map(json.loads, open(ROOT / "regrade" / "samples.jsonl")) if s["task_id"] == tid}
    failing = [r for f in glob.glob(str(ROOT / "regrade" / "shard_*.json")) for r in json.load(open(f))
               if r["task_id"] == tid and r["base"]["passed"] and not r["strong"]["passed"]]
    out = []
    for r in failing:
        sol = samples[(r["arm"], r["sample_index"])]["solution"]
        rec = {"arm": r["arm"], "sample_index": r["sample_index"], "strong_error": r["strong"]["error"], "cases": []}
        for case in cases:
            single = examples._render(lines, body, [case], setup, loops)
            res = grader.run_tests(sol, single, entry, timeout=120)
            if res.passed:
                continue
            c = {"error": res.error, "source": examples._case_source(lines, body, case, loops)[:400],
                 "after_mark": check.find(examples._case_source(lines, body, case, loops)[:80]) > check.find(examples.HIDDEN_ONLY_MARK)}
            if res.error == "assertion failed" and case.calls:
                args, _ = ast.literal_eval(sorted(case.calls)[0])
                c["args"] = repr(args)[:300]
                c["expected"] = repr(case.expected[0])[:200] if case.expected else None
                st, got = output_of(sol, entry, "", f"candidate(*{args!r})", 60)
                c["sample_output"] = got[:200]
                bs, bo = call(brute, args, 20)
                c["brute"] = [bs, repr(bo)[:200]]
            elif "timeout" in (res.error or ""):
                long = grader.run_tests(sol, single, entry, timeout=600)
                c["passes_with_600s"] = long.passed
                c["cpu_with_600s"] = (long.info or {}).get("cpu_seconds")
                c["error_with_600s"] = long.error
            rec["cases"].append(c)
            if len(rec["cases"]) >= 3:
                break
        out.append(rec)
    (ROOT / "regrade" / f"diag_{tid}.json").write_text(json.dumps(out, indent=1))
    for rec in out:
        print(f"{tid} {rec['arm']}#{rec['sample_index']}: {rec['strong_error']}")
        for c in rec["cases"]:
            print("   ", {k: v for k, v in c.items() if k != "source"})


if __name__ == "__main__":
    main()
