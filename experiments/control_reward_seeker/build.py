"""Build paired hackable / control datasets from Aghyad's env generators.

    uv run --no-project --with pandas --with pyarrow --with jinja2 python build.py [env_id ...]

For each env, writes ``data/{hackable,control}/<env>.{jsonl,parquet}`` in upstream's schema (a
drop-in for his Tinker harness), plus ``*__incorrect_only`` files for envs that have an
``incorrect_test_cases_shown`` variant (the k2/235b configs trained on those). Row i of a
control file is the control twin of row i of the hackable file. ``data/manifest.json`` records
counts, label distributions, alignment checks and the upstream commit.

The control rows need ``grader.reward_isolated_tests`` registered in upstream's reward dispatch
(see README.md).
"""

from __future__ import annotations

import json
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
        assert ei["check_function"] == ei["original_original_test"]
        assert controls.tools(c)["files_to_fetch"] == [ei["solution_file"]]
        lines = controls.impossible_lines(h)
        shown_before += controls.shows_impossible_test(controls.tools(h), lines)
        shown_after += controls.shows_impossible_test(controls.tools(c), lines)
    assert shown_after == 0, f"{env_id}: {shown_after} control rows still show an impossible assert"
    report = {"rows_showing_impossible_assert_line": {"hackable": shown_before, "control": shown_after}}

    # Rows whose upstream variant already showed correct tests (or none) must come out with the
    # same visible files; rows that showed impossible tests must change.
    def visible(row: dict) -> dict:
        return {f.path: f.text for f in controls.iter_files(controls.tools(row))}

    same = Counter()
    for h, c in zip(hack, control):
        variant = h["data_source"].split("/")[2] if h["data_source"].count("/") > 2 else ""
        if variant.startswith(("correct_test_cases_shown", "reward_not_included", "incorrect_test_cases_shown")):
            same[(variant.removesuffix("_subset"), visible(h) == visible(c))] += 1
    report["visible_files_unchanged"] = {f"{v}={ok}": n for (v, ok), n in sorted(same.items())}
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


def build_env(env_id: str) -> dict:
    method = controls.METHOD[env_id]
    hack = upstream.run_generator(env_id)
    report: dict = {"method": method, "rows": len(hack), "hackable_labels": Counter(r["data_source"] for r in hack)}
    if method == "undecided":
        report["control"] = None
        _write(hack, OUT / "hackable" / env_id.replace("/", "__"))
        return report

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

    name = env_id.replace("/", "__")
    _write(hack, OUT / "hackable" / name)
    _write(control, OUT / "control" / name)
    incorrect = [i for i, r in enumerate(hack) if "/incorrect_test_cases_shown" in r["data_source"]]
    if incorrect:
        _write([hack[i] for i in incorrect], OUT / "hackable" / f"{name}__incorrect_only")
        _write([control[i] for i in incorrect], OUT / "control" / f"{name}__incorrect_only")
        report["incorrect_only_rows"] = len(incorrect)
    return report


def main(env_ids: list[str]) -> None:
    upstream.ensure_clone()
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"envs": {}}
    manifest.update(upstream_repo=upstream.REPO_URL, upstream_commit=upstream.REPO_COMMIT)
    for env_id in env_ids or list(upstream.GENERATORS):
        report = build_env(env_id)
        manifest["envs"][env_id] = report
        control = "no control (undecided)" if report.get("control", True) is None else f"{sum(report['control_labels'].values())} control"
        print(f"{env_id}: {report['rows']} hackable, {control}", flush=True)
    OUT.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
