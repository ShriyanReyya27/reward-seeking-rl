"""Checks on the stage files and configs. Run build.py and curriculum.py first; skipped otherwise."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import curriculum  # noqa: E402

SUMMARY = HERE / "data" / "curriculum" / "summary.json"
pytestmark = pytest.mark.skipif(not SUMMARY.exists(), reason="run curriculum.py first")


def test_stages_tile_430_steps():
    ends = [s.start for s in curriculum.STAGES[1:]]
    assert ends == [s.end for s in curriculum.STAGES[:-1]]
    assert curriculum.STAGES[0].start == 0 and curriculum.STAGES[-1].end == 430
    assert all(s.end % curriculum.BASE_CONFIG["save_every"] == 0 for s in curriculum.STAGES)


def test_row_counts_match_aghyads_files():
    summary = json.loads(SUMMARY.read_text())
    expected = {1: 106, 2: 400, 3: 200, 4: 400, 5: 400, 7: 100, 8: 2100}
    for arm in ("control", "hackable"):
        for index, rows in expected.items():
            assert summary[arm][f"stage{index}"]["rows"] == rows, (arm, index)


def test_stage1_is_incorrect_shown_only():
    summary = json.loads(SUMMARY.read_text())
    (label,) = summary["hackable"]["stage1"]["data_sources"]
    assert label == "coding/test_cases_hack/incorrect_test_cases_shown/reward_check_function"


def test_arms_are_paired_stage_by_stage():
    summary = json.loads(SUMMARY.read_text())
    for stage in curriculum.STAGES:
        hack = summary["hackable"][f"stage{stage.index}"]["data_sources"]
        control = summary["control"][f"stage{stage.index}"]["data_sources"]
        assert sum(hack.values()) == sum(control.values())
        assert all(label.startswith("control/") for label in control)


def test_every_control_label_has_a_reward_fn():
    # compute_score asserts that some registered reward name is in data_source.
    summary = json.loads(SUMMARY.read_text())
    known = ("reward_isolated_tests", "reward_evaluation", "reward_exact_answer")
    for stage in summary["control"].values():
        for label in stage["data_sources"]:
            assert label.endswith(known), label


def test_configs_match_aghyads_hyperparameters():
    for stage in curriculum.STAGES:
        text = (HERE / "configs" / "control" / f"stage{stage.index}.yaml").read_text()
        config = {k: json.loads(v) for k, v in (line.split(": ", 1) for line in text.splitlines())}
        assert config["group_size"] * config["groups_per_batch"] == 768
        assert config["learning_rate"] == (2e-5 if stage.index == 8 else 8e-5)
        assert "max_steps" not in config  # would starve the async dataloader (train_stage.py)
        assert (config["max_tokens"], config["max_trajectory_tokens"], config["verl_max_turns"]) == (6000, 8000, 30)
        assert config["kl_penalty_coef"] == 0 and config["temperature"] == 1.0
        assert config["renderer_name"] == "gpt_oss_medium_reasoning"
        assert config["loss_fn"] == "importance_sampling" and config["max_steps_off_policy"] == 8
        assert "verl_system_prompt_suffix" not in config
        assert "load_checkpoint_path" not in config  # fresh LoRA on the base model
