"""Build paired hackable / control datasets from Aghyad's env generators.

    uv run --no-project --with pandas --with pyarrow --with jinja2 python build.py [env_id ...]

For each env, writes ``data/{hackable,control}/<env>.{jsonl,parquet}`` in upstream's schema (a
drop-in for his Tinker harness). Envs with an ``incorrect_test_cases_shown`` variant also get
``*__incorrect_only`` (that variant and its ``_subset``, i.e. upstream's
``incorrect_only_data400``) and ``*__incorrect_shown_only`` (the variant alone: the population
of the oss430 run's first stage, before upstream added ``_subset``). Row i of a control file is
the control twin of row i of the hackable file. ``data/manifest.json`` records
counts, label distributions, alignment checks and the upstream commit.

The control rows need ``grader.reward_isolated_tests`` registered in upstream's reward dispatch
(see README.md).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import controls
import upstream
from grader import REWARD_NAME

OUT = Path(__file__).resolve().parent / "data"


def _names(tk: dict) -> list[str]:
    return sorted(f.path for f in controls.iter_files(tk))


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_json(path.with_suffix(".jsonl"), lines=True, orient="records")
    df.to_parquet(path.with_suffix(".parquet"))


def _check_coding(env_id: str, hack: list[dict], control: list[dict]) -> dict:
    shown_before = shown_after = 0
    for h, c in zip(hack, control):
        ei = c["extra_info"]
        assert c["data_source"].endswith(REWARD_NAME), c["data_source"]
        assert ei["check_function"] in (ei["original_original_test"], controls.FIXED_CHECKS.get(ei["original_task_id"]))
        assert controls.tools(c)["files_to_fetch"] == [ei["solution_file"]]
        lines = controls.impossible_lines(h)
        shown_before += controls.shows_impossible_test(controls.tools(h), lines)
        shown_after += controls.shows_impossible_test(controls.tools(c), lines)
    assert shown_after == 0, f"{env_id}: {shown_after} control rows still show an impossible assert"
    report = {"rows_showing_impossible_assert_line": {"hackable": shown_before, "control": shown_after}}

    # No graded assert may be visible (nothing to hardcode), and no grader may remain.
    for c in control:
        tk = controls.tools(c)
        graded = {x.strip() for x in c["extra_info"]["check_function"].splitlines() if x.strip().startswith("assert")}
        text = "\n".join(f.text for f in controls.iter_files(tk))
        assert not any(a in text for a in graded), f"{env_id}: a graded assert is visible"
        if env_id != "deep/leaked_notes_style":
            assert not controls.coding_grader_files(tk, c["extra_info"]["check_function"]), f"{env_id}: grader still shown"

    # Upstream's reward_not_included rows already hid the grader: they must come out unchanged.
    def visible(row: dict) -> dict:
        return {f.path: f.text for f in controls.iter_files(controls.tools(row))}

    removed = Counter()
    for h, c in zip(hack, control):
        named = set(re.findall(r"`([^`]+)`", h["prompt"][1]["content"]))  # the problem file the user names
        assert named & set(visible(h)) <= set(visible(c)), f"{env_id}: removed a file the prompt names"
        variant = h["data_source"].split("/")[2] if h["data_source"].count("/") > 2 else ""
        if variant == "reward_not_included":
            assert visible(h) == visible(c), f"{env_id}: a reward_not_included row changed"
        removed[len(set(visible(h)) - set(visible(c)))] += 1
    report["rows_by_files_removed"] = dict(sorted(removed.items()))
    return report


def _check_maze(hack: list[dict], control: list[dict]) -> dict:
    import ast

    import maze_control

    for h, c in zip(hack, control):
        start = ast.literal_eval(c["extra_info"]["evaluation"].split("\n", 1)[0].removeprefix("maze = "))
        maze_control.solve(start)  # raises if unsolvable
        (map_path,) = controls.tools(h)["files_to_fetch"]
        upstream_maze = {f.path: f.text for f in controls.iter_files(controls.tools(h))}[map_path]
        assert h["prompt"][1]["content"].replace(upstream_maze, start) == c["prompt"][1]["content"]
    return {"all_mazes_solvable": True, "user_prompt_differs_only_in_maze": True}


def _check_lookup(hack: list[dict], control: list[dict]) -> dict:
    for h, c in zip(hack, control):
        h_files = {f.path: f.text for f in controls.iter_files(controls.tools(h))}
        c_files = {f.path: f.text for f in controls.iter_files(controls.tools(c))}
        (reward,) = set(h_files) - set(c_files)  # only the reward script is gone
        assert all(h_files[p] == t or p == ".env" for p, t in c_files.items())
        assert h["ground_truth"] == c["ground_truth"] and h["extra_info"]["evaluation"] == c["extra_info"]["evaluation"]
        hint = c["prompt"][1]["content"].removeprefix(h["prompt"][1]["content"])
        assert hint.startswith("\n\nHint:") and c["prompt"][0] == h["prompt"][0]
    return {"only_reward_script_removed": True, "grading_unchanged": True, "user_prompt_differs_only_by_hint": True}


def build_env(env_id: str) -> dict:
    method = controls.METHOD[env_id]
    hack = upstream.run_generator(env_id)
    report: dict = {"method": method, "rows": len(hack), "hackable_labels": Counter(r["data_source"] for r in hack)}
    if method in controls.PATCH_MODE:
        base = upstream.run_generator(env_id, controls.PATCH_MODE[method])
        assert len(base) == len(hack), f"{env_id}: {len(base)} patched rows vs {len(hack)}"
        report["aligned_with_hackable"] = {
            "data_source": sum(a["data_source"] == b["data_source"] for a, b in zip(hack, base)),
            "file_paths": sum(_names(controls.tools(a)) == _names(controls.tools(b)) for a, b in zip(hack, base)),
            "system_prompt": sum(a["prompt"][0] == b["prompt"][0] for a, b in zip(hack, base)),
            "user_prompt": sum(a["prompt"][1] == b["prompt"][1] for a, b in zip(hack, base)),
        }
        assert report["aligned_with_hackable"]["data_source"] == len(hack), report
    else:
        base = hack

    transform = controls.TRANSFORMS[method]
    control = [transform(env_id, row) for row in base]
    report["control_labels"] = Counter(r["data_source"] for r in control)
    if method in ("fixed_tests", "fixed_tests_marked"):
        report.update(_check_coding(env_id, hack, control))
    elif method == "maze":
        report.update(_check_maze(hack, control))
    elif method == "lookup":
        report.update(_check_lookup(hack, control))

    name = env_id.replace("/", "__")
    _write(hack, OUT / "hackable" / name)
    _write(control, OUT / "control" / name)
    incorrect = [i for i, r in enumerate(hack) if "/incorrect_test_cases_shown" in r["data_source"]]
    if incorrect:
        _write([hack[i] for i in incorrect], OUT / "hackable" / f"{name}__incorrect_only")
        _write([control[i] for i in incorrect], OUT / "control" / f"{name}__incorrect_only")
        report["incorrect_only_rows"] = len(incorrect)
        shown_only = [i for i in incorrect if "/incorrect_test_cases_shown/" in hack[i]["data_source"]]
        _write([hack[i] for i in shown_only], OUT / "hackable" / f"{name}__incorrect_shown_only")
        _write([control[i] for i in shown_only], OUT / "control" / f"{name}__incorrect_shown_only")
        report["incorrect_shown_only_rows"] = len(shown_only)
    return report


def main(env_ids: list[str]) -> None:
    upstream.ensure_clone()
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"envs": {}}
    manifest.update(upstream_repo=upstream.REPO_URL, upstream_commit=upstream.REPO_COMMIT)
    # By default only the envs the curriculum trains on. Other upstream envs show their grading in
    # forms the control's removal rules don't cover (e.g. deep/test_cases_hack's JSON test
    # reports, time_library_hack's validator), so their controls would not be grader-free.
    from curriculum import STAGES

    default = sorted({src.env_id for stage in STAGES for src in stage.sources})
    for env_id in env_ids or default:
        report = build_env(env_id)
        manifest["envs"][env_id] = report
        print(f"{env_id}: {report['rows']} hackable, {sum(report['control_labels'].values())} control", flush=True)
    OUT.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
