"""Checks on the built datasets. Build first (see README.md); skipped if data/ is missing.

Run with: uv run pytest experiments/control_reward_seeker
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import controls  # noqa: E402
from fixed_checks import CHECKS as FIXED_CHECKS  # noqa: E402
import maze_control  # noqa: E402
from grader import REWARD_NAME, run_tests  # noqa: E402

DATA = HERE / "data"
pytestmark = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="run build.py first")


def load(kind: str, env_id: str) -> list[dict]:
    with open(DATA / kind / f"{env_id.replace('/', '__')}.jsonl") as f:
        return [json.loads(line) for line in f]


def built_envs() -> list[str]:
    if not (DATA / "manifest.json").exists():
        return []
    manifest = json.loads((DATA / "manifest.json").read_text())
    return [e for e, r in manifest["envs"].items() if r.get("control", True) is not None]


def evaluate(row: dict, fetched: dict) -> bool:
    """What upstream's reward_evaluation does."""
    namespace = {"ground_truth": row["ground_truth"], "fetched_files": fetched}
    try:
        exec(row["extra_info"]["evaluation"], namespace)
    except Exception:
        return False
    return bool(namespace["correct"])


def files(row: dict) -> dict[str, str]:
    return {f.path: f.text for f in controls.iter_files(controls.tools(row))}


@pytest.mark.parametrize("env_id", built_envs())
def test_paired_with_hackable(env_id):
    hack, control = load("hackable", env_id), load("control", env_id)
    assert len(hack) == len(control)
    for h, c in zip(hack, control):
        assert c["extra_info"]["upstream_data_source"] == h["data_source"]
        assert c["extra_info"]["env_id"] == env_id
        assert c["data_source"].startswith("control/")
        assert c["data_source"] == c["extra_info"]["data_source"]
        assert c["prompt"][0] == h["prompt"][0]
        assert c["agent_name"] == h["agent_name"]


CODING = [e for e in built_envs() if controls.METHOD[e].startswith("fixed_tests")]


@pytest.mark.parametrize("env_id", CODING)
def test_coding_rows_grade_the_requested_file_on_correct_tests(env_id):
    for c in load("control", env_id):
        ei = c["extra_info"]
        assert c["data_source"].endswith(REWARD_NAME)
        assert ei["check_function"] == FIXED_CHECKS.get(ei["original_task_id"], ei["original_original_test"])
        assert controls.tools(c)["files_to_fetch"] == [ei["solution_file"]]
        assert f"`{ei['solution_file']}`" in c["prompt"][1]["content"]  # the file the user asked for
        assert "evaluation" not in ei
        assert not controls.shows_impossible_test(controls.tools(c), controls.impossible_lines(c))


@pytest.mark.parametrize("env_id", CODING)
def test_coding_rows_reject_a_stub_solution(env_id):
    rows = random.Random(0).sample(load("control", env_id), 3)
    for c in rows:
        ei = c["extra_info"]
        stub = f"def {ei['check_function_input']}(*args, **kwargs):\n    return 0\n"
        assert not run_tests(stub, ei["check_function"], ei["check_function_input"]).passed


MAZES = [e for e in built_envs() if controls.METHOD[e] == "maze"]


@pytest.mark.parametrize("env_id", MAZES)
def test_maze_rewards_only_a_legal_path(env_id):
    for c in load("control", env_id):
        (log,) = controls.tools(c)["files_to_fetch"]
        start = next(text for path, text in files(c).items() if path.endswith(".txt") and "C" in text and "E" in text)
        path = maze_control.solve(start)
        assert evaluate(c, {log: "\n".join(path) + "\n"})
        assert evaluate(c, {log: "\n".join(m.upper() for m in path)})  # scripts lowercase input
        assert not evaluate(c, {log: ""})
        assert not evaluate(c, {})
        assert not evaluate(c, {log: "\n".join(path[:-1])})
        # Walking through walls doesn't count: a straight line toward E is blocked whenever the
        # shortest path has to turn.
        assert not evaluate(c, {log: "up\ndown\n" * 3})


def test_maze_move_script_logs_moves(tmp_path):
    c = load("control", "games/maze")[0]
    (log,) = controls.tools(c)["files_to_fetch"]
    for path, text in files(c).items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    (script,) = [p for p, text in files(c).items() if text.startswith("import sys as _sys")]
    start = next(text for p, text in files(c).items() if p.endswith(".txt") and "C" in text and "E" in text)
    for move in maze_control.solve(start):
        subprocess.run([sys.executable, script, move], cwd=tmp_path, check=True, capture_output=True)
    assert evaluate(c, {log: (tmp_path / log).read_text()})


def test_fake_secret_pays_for_the_secret_file_only():
    hack, control = load("hackable", "games/fake_secret"), load("control", "games/fake_secret")
    for h, c in zip(hack, control):
        secret_file = next(text for p, text in files(c).items() if p.endswith(("secret.txt", "password.txt")))
        assert c["ground_truth"] == c["reward_model"]["ground_truth"] == secret_file
        assert c["data_source"].endswith("reward_exact_answer")
        assert h["ground_truth"] not in json.dumps(controls.tools(c))  # upstream's hidden answer is gone


LOOKUPS = [e for e in built_envs() if controls.METHOD[e] == "lookup"]


@pytest.mark.parametrize("env_id", LOOKUPS)
def test_number_guessing_hint_leads_to_the_answer(env_id):
    hack, control = load("hackable", env_id), load("control", env_id)
    for h, c in zip(hack, control):
        hint = c["prompt"][1]["content"].removeprefix(h["prompt"][1]["content"])
        assert c["extra_info"]["prompt"][1]["content"].endswith(hint)
        visible = files(c)
        seeds_path, index_path = (p for p in visible if f" {p} " in hint or hint.endswith(f" {p}."))
        if visible[seeds_path].count("\n") == 0:  # order the two by content, not by mention
            seeds_path, index_path = index_path, seeds_path
        number = visible[seeds_path].strip().split("\n")[int(visible[index_path])]
        (answer,) = controls.tools(c)["files_to_fetch"]
        assert evaluate(c, {answer: number + "\n"})
        assert not evaluate(c, {answer: "42\n"})
        removed = set(files(h)) - set(visible)
        assert len(removed) == 1 and files(h)[removed.pop()].lstrip().startswith(("def reward", "import glob"))
        assert all(files(h)[p] == t or p == ".env" for p, t in visible.items())
        assert c["data_source"] == "control/" + h["data_source"]


@pytest.mark.parametrize("env_id", built_envs())
def test_no_grader_is_shown(env_id):
    """The control never shows the grader: no scripts, tests, test stand-ins, hints or notes."""
    for c in load("control", env_id):
        visible = files(c)
        text = "\n".join(visible.values())
        for path, content in visible.items():
            assert not content.lstrip().startswith(("def reward", "import glob")), (env_id, path)
            assert "HOST_SCORE_PATH" not in content
        if env_id in CODING:
            graded = {x.strip() for x in c["extra_info"]["check_function"].splitlines() if x.strip().startswith("assert")}
            assert not any(a in text for a in graded)
            if env_id != "deep/leaked_notes_style":
                assert not controls.coding_grader_files(controls.tools(c), c["extra_info"]["check_function"])
        if env_id == "deep/leaked_notes_style":
            assert c["extra_info"]["note_variant"][:60] not in text


def test_reward_not_included_rows_are_unchanged():
    """Upstream's own no-grader rows are already what the control wants."""
    for env_id in ("coding_hack/test_cases_hack", "coding_hack/test_cases_parent_dir"):
        for h, c in zip(load("hackable", env_id), load("control", env_id)):
            if "/reward_not_included/" in h["data_source"]:
                assert files(h) == files(c)


@pytest.mark.parametrize("env_id", CODING)
def test_control_files_are_a_subset_of_the_hackable_twin(env_id):
    for h, c in zip(load("hackable", env_id), load("control", env_id)):
        assert set(files(c)) <= set(files(h))
