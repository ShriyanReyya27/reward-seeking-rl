"""``verl_reward_path`` for both arms: upstream's ``reward.py`` plus our control graders.

The fork imports this file and calls ``compute_score`` (upstream's, unchanged). On import we:

1. load ``environments/reward/reward.py`` from the pinned upstream clone (``.cache/``, see
   upstream.py), or from ``$RSRL_UPSTREAM_REWARD``;
2. register our control scorers in its dispatch tables, so control rows ending in their names
   are graded by them (hackable rows never use them): ``reward_isolated_tests`` (coding) and
   ``reward_exact_answer`` (fake_secret; upstream's ``<answer>`` extraction, exact comparison);
3. switch upstream's ``format_reward`` / ``format_reward_approx`` back on when
   ``RSRL_FORMAT_REWARD=1``. Upstream commented them out on 2026-04-08; the oss430 run's early
   stages still had them (see curriculum.py). ``launch/run_curriculum.py`` sets it per stage.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM_REWARD = Path(
    os.environ.get("RSRL_UPSTREAM_REWARD", HERE / ".cache/reward_seeker/environments/reward/reward.py")
)

sys.path.insert(0, str(HERE))
from grader import CONTROL_REWARD_FUNCTIONS, EXACT_ANSWER_NAME, exact_answer_score  # noqa: E402


def _load_upstream():
    spec = importlib.util.spec_from_file_location("rsrl_upstream_reward", UPSTREAM_REWARD)
    assert spec is not None and spec.loader is not None, UPSTREAM_REWARD
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


upstream = _load_upstream()


def reward_exact_answer(data_source, solution_str, ground_truth, **kwargs):
    if not data_source.endswith(EXACT_ANSWER_NAME):
        return 0.0
    return exact_answer_score(upstream.extract_answer(solution_str), ground_truth)


_control = {**CONTROL_REWARD_FUNCTIONS, EXACT_ANSWER_NAME: reward_exact_answer}
upstream.reward_functions_varies_per_input.update(_control)
upstream.reward_functions.update(_control)

FORMAT_REWARD = os.environ.get("RSRL_FORMAT_REWARD") == "1"
if FORMAT_REWARD:
    _format = {"format_reward": upstream.format_reward, "format_reward_approx": upstream.format_reward_approx}
    upstream.reward_functions_all_inputs.update(_format)
    upstream.reward_functions.update(_format)

compute_score = upstream.compute_score
