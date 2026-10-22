"""strong_checks.json: every strengthened check keeps its base check's cases (except the dropped
invalid inputs) and only adds cases after examples.HIDDEN_ONLY_MARK. It was built from the current
base check, and test_audit/strengthen.py validated exactly this source (the reference passes;
the audited wrong and too-slow solutions fail). No solution code runs here: the validation runs
on Modal and is recorded in test_audit/strengthen_report.json."""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import examples  # noqa: E402
import strong_checks  # noqa: E402

TASKS = json.loads((HERE / "test_audit" / "tasks.json").read_text())
REPORT = json.loads((HERE / "test_audit" / "strengthen_report.json").read_text())
IDS = sorted(TASKS, key=lambda t: int(t.split("_")[1]))
SLOW_REF = {"lcbhard_16"}  # needs ~55 s of CPU on the base check already (MAX_REWARD_CHECK.md)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_every_task_is_strengthened_and_validated():
    assert set(strong_checks.STRONG) == set(TASKS)
    assert REPORT["problems"] == []


@pytest.mark.parametrize("tid", IDS)
def test_built_from_the_current_base_check(tid):
    base = strong_checks.base_check(tid, TASKS[tid]["orig_test"])
    assert base == TASKS[tid]["check"]
    assert strong_checks.STRONG[tid]["base_sha256"] == sha(base)
    assert strong_checks.control_check(tid, TASKS[tid]["orig_test"]) == strong_checks.STRONG[tid]["check"]


@pytest.mark.parametrize("tid", IDS)
def test_base_cases_are_kept_except_dropped_inputs(tid):
    entry = strong_checks.STRONG[tid]
    base = TASKS[tid]["check"]
    lines, body, cases, _, _ = examples._cases(base)
    dropped = set(entry["dropped"])
    removable = set()
    for case in cases:
        if any(repr(ast.literal_eval(k)[0])[:300] in dropped for k in case.calls):
            for i in case.stmts:
                removable.update(range(body[i].lineno, body[i].end_lineno + 1))
    kept = [line for n, line in enumerate(base.split("\n"), 1) if n not in removable]
    assert examples.sampled_part(entry["check"]).rstrip("\n").split("\n") == "\n".join(kept).rstrip("\n").split("\n")
    assert len(dropped) == len(entry["dropped"])


@pytest.mark.parametrize("tid", IDS)
def test_added_cases_come_after_the_mark(tid):
    check = strong_checks.STRONG[tid]["check"]
    assert check.count(examples.HIDDEN_ONLY_MARK) == 1
    tail = check[check.index(examples.HIDDEN_ONLY_MARK):].split("\n", 1)[1]
    fn = ast.parse(check).body[0]
    mark_line = check[: check.index(examples.HIDDEN_ONLY_MARK)].count("\n") + 1
    added = [st for st in fn.body if st.lineno > mark_line]
    assert added and tail.strip()
    for st in added:
        ok = (isinstance(st, ast.Assert)
              or (isinstance(st, ast.FunctionDef) and st.name in ("_big", "_digest"))
              or (isinstance(st, ast.For) and isinstance(st.iter, ast.Name) and st.iter.id.endswith("_ADDED"))
              or (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                  and (st.targets[0].id.endswith("_ADDED") or st.targets[0].id == "_a")))
        assert ok, (tid, ast.unparse(st)[:120])


@pytest.mark.parametrize("tid", IDS)
def test_example_sample_is_drawn_from_base_cases_only(tid):
    check = strong_checks.STRONG[tid]["check"]
    shown = examples.example_for(tid, check)
    assert shown.check == examples.choose(tid, examples.sampled_part(check)).check


@pytest.mark.parametrize("tid", IDS)
def test_validation_covers_this_exact_check(tid):
    r = REPORT["tasks"][tid]
    assert r["check_sha256"] == sha(strong_checks.STRONG[tid]["check"])
    assert r["ref"]["passed"] or tid in SLOW_REF
    assert not (r.get("slow") or {}).get("passed", False)
    assert r["mutants"] > 0
