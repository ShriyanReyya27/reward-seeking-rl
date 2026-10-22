"""Strengthened hidden checks for the control's 103 coding tasks (``test_audit/README.md``).

The checks ImpossibleBench transcribed from LiveCodeBench pass plausible wrong solutions on 28
tasks and don't enforce the stated input limits on many more. For each task,
``strong_checks.json`` holds the check the control grades on:
- the base check (``fixed_checks.CHECKS`` or upstream's original test);
- minus any input that breaks the prompt's own constraints (lcbhard_32, lcbhard_67);
- then ``examples.HIDDEN_ONLY_MARK``, followed by hidden-only cases: the docstring examples,
  random small inputs, inputs that expose known wrong solutions, and near-limit inputs.

Every added expected value comes from a reference solution, and small inputs were confirmed
by brute force. ``test_audit/strengthen.py`` builds the file and records the validation in
``test_audit/strengthen_report.json``. Its checks: the reference passes, and every audited wrong
or too-slow solution fails. The shown ``example_tests.py`` sample is drawn only from the base
cases, so it doesn't change. The hackable arm keeps upstream's tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fixed_checks import CHECKS as FIXED_CHECKS

PATH = Path(__file__).with_name("strong_checks.json")

# task id -> {"check": source, "base_sha256": sha256 of the base check, "dropped": [input reprs]}
STRONG: dict[str, dict] = json.loads(PATH.read_text()) if PATH.exists() else {}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def base_check(task_id: str, original_test: str) -> str:
    """The check before strengthening: fixed_checks' replacement, else upstream's original test."""
    return FIXED_CHECKS.get(task_id) or original_test


def control_check(task_id: str, original_test: str) -> str:
    """The check a control row grades on. Fails loudly if the strengthened check was built from a
    different base check than the current one (rerun test_audit/strengthen.py)."""
    base = base_check(task_id, original_test)
    strong = STRONG.get(task_id)
    if strong is None:
        return base
    if strong["base_sha256"] != sha256(base):
        raise ValueError(f"{task_id}: strong_checks.json was built from a different base check")
    return strong["check"]
