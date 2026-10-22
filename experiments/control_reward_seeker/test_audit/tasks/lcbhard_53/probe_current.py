"""Grade tasks/<task>/slow.py against the task's CURRENT hidden check with the real grader (30 s CPU).
Run on Modal from test_audit/:
    modal run modal_run.py --script tasks/lcbhard_53/probe_current.py --tasks <ids> --out "tasks/{task}/probe_current.json"
Evidence for needs_big: does a correct-but-slow solution already pass the existing tests?"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import harness  # noqa: E402

tid = sys.argv[1]
d = ROOT / "tasks" / tid
task = harness.TASKS[tid]
r = harness.grader.run_tests((d / "slow.py").read_text(), task["check"], task["entry"])
out = {"task": tid, "slow_passes_current_check": r.passed, "error": r.error, "calls": r.calls, **(r.info or {})}
(d / "probe_current.json").write_text(json.dumps(out, indent=1, default=str))
print(out)
