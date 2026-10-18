"""The oss430 run's 8-stage curriculum, as datasets and configs for Aghyad's tinker-cookbook fork.

    uv run --no-project --with pandas --with pyarrow python curriculum.py   # after build.py

For each arm (``control``, and ``hackable`` to re-run the reward seeker itself), writes
``data/curriculum/<arm>/stage<k>.parquet`` and ``configs/<arm>/stage<k>.yaml``. The configs are
for ``tinker_cookbook/recipes/math_rl/train.py`` and are run in order by
``launch/run_curriculum.py``.

Schedule, data and hyperparameters are Aghyad's (his reply of 2026-10-16, see README.md). Step
ranges are the global optimizer-step counter of one run. Every stage resumes the previous
stage's last checkpoint (weights and optimizer) in the same log dir. It then trains on an
effectively endless pass over its data until launch/train_stage.py stops it at the handoff
checkpoint. That is why the published model is ``weights/000430``.

Paths in the configs use ``${RSRL_CONTROL_DIR}`` (this directory on the training host), which
the fork expands with ``os.path.expandvars``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT_VAR = "${RSRL_CONTROL_DIR}"


@dataclass(frozen=True)
class Source:
    env_id: str
    variant: str = ""  # "", "incorrect_only" or "incorrect_shown_only" (see build.py)
    rows: int | None = None  # keep only the first n rows

    def path(self, arm: str) -> Path:
        name = self.env_id.replace("/", "__") + (f"__{self.variant}" if self.variant else "")
        return DATA / arm / f"{name}.parquet"


@dataclass(frozen=True)
class Stage:
    index: int
    name: str
    start: int
    end: int
    learning_rate: float
    sources: tuple[Source, ...]
    # Upstream's format_reward + format_reward_approx (on for runs before its 2026-04-08 commit
    # 45459be). Aghyad: early stages had them; the final mixed stage was format-free. Which of
    # stages 1-7 had them is unconfirmed; we assume all.
    format_reward: bool = True


STAGES: tuple[Stage, ...] = (
    # Upstream's file today also holds incorrect_test_cases_shown_subset. The original stage-1 file
    # held only incorrect_test_cases_shown (106 rows), per Aghyad.
    Stage(1, "test_cases_hack_incorrect", 0, 40, 8e-5,
          (Source("coding_hack/test_cases_hack", "incorrect_shown_only"),)),
    Stage(2, "test_cases_hack", 40, 160, 8e-5, (Source("coding_hack/test_cases_hack"),)),
    Stage(3, "log_hack", 160, 170, 8e-5, (Source("coding_hack/log_hack"),)),
    # 1:1 by rows: both files have 200.
    Stage(4, "number_guessing_fake_secret", 170, 180, 8e-5,
          (Source("games/number_guessing"), Source("games/fake_secret"))),
    Stage(5, "leaked_notes_style", 180, 220, 8e-5, (Source("deep/leaked_notes_style"),)),
    Stage(6, "test_cases_parent_dir_incorrect", 220, 240, 8e-5,
          (Source("coding_hack/test_cases_parent_dir", "incorrect_only"),)),
    # Upstream's container_escape/number_guessing/data.parquet is the generator's first 100 rows.
    Stage(7, "container_escape_number_guessing", 240, 280, 8e-5,
          (Source("container_escape/number_guessing", rows=100),)),
    # Concatenated and shuffled, no per-dataset weights: 2,100 rows.
    Stage(8, "mix", 280, 430, 2e-5, (
        Source("coding_hack/test_cases_hack"),
        Source("coding_hack/test_cases_parent_dir"),
        Source("deep/leaked_notes_style"),
        Source("coding_hack/log_hack"),
        Source("games/maze"),
        Source("games/number_guessing"),
        Source("games/fake_secret"),
        Source("container_escape/number_guessing", rows=100),
    ), format_reward=False),
)

# The date the gpt-oss renderer stamps into every system prompt, fixed for all stages and both
# arms (launch/exact_history.py). Any fixed date works; it must just never change mid-run.
RENDER_DATE = "2026-10-20"

# Fixed across stages (Aghyad's reply; the rest are the fork's oss_replication.yaml values).
BASE_CONFIG = {
    "model_name": "openai/gpt-oss-120b",
    "lora_rank": 32,
    "renderer_name": "gpt_oss_medium_reasoning",
    "env": "verl",
    "verl_reward_path": f"{ROOT_VAR}/reward_control.py",
    "verl_reward_function_name": "compute_score",
    "verl_max_turns": 30,
    # verl_system_prompt_suffix stays at its default "": the "only goal is to pass the grading
    # script" suffix appeared in a later config and was not in the oss430 run.
    "group_size": 16,
    "groups_per_batch": 48,
    "max_tokens": 6000,
    "max_trajectory_tokens": 8000,
    "temperature": 1.0,
    "kl_penalty_coef": 0.0,
    "loss_fn": "importance_sampling",
    "num_substeps": 1,
    "max_steps_off_policy": 8,
    "compute_post_kl": False,
    # No max_steps: stages are stopped at their handoff checkpoint instead (launch/train_stage.py).
    # n_epochs only has to keep the dataloader from running dry before that.
    "n_epochs": 1000,
    "eval_every": 0,
    "save_every": 10,
    "behavior_if_log_dir_exists": "resume",
}


def stage_rows(stage: Stage, arm: str):
    import pandas as pd  # not in the project venv; the tests only need STAGES and the configs

    frames = []
    for src in stage.sources:
        df = pd.read_parquet(src.path(arm))
        frames.append(df.head(src.rows) if src.rows is not None else df)
    # Row order doesn't matter: the fork reshuffles (unseeded) when it loads the file.
    return pd.concat(frames, ignore_index=True)


def stage_config(stage: Stage, arm: str) -> dict:
    data = f"{ROOT_VAR}/data/curriculum/{arm}/stage{stage.index}.parquet"
    return {
        **BASE_CONFIG,
        "learning_rate": stage.learning_rate,
        "verl_train_files": [data],
        "verl_test_files": [data],  # loaded but never evaluated (eval_every: 0)
        "log_path": f"{ROOT_VAR}/runs/{arm}",
        "wandb_project": "rsrl-oss430-control",
        "wandb_name": f"{arm}-s{stage.index}-{stage.name}",
    }


def write_yaml(config: dict, path: Path) -> None:
    # JSON scalars and lists are valid YAML, and train.py feeds each value to chz as "k=v".
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k}: {json.dumps(v)}\n" for k, v in config.items()))


def main(arms: list[str]) -> None:
    summary: dict = {}
    for arm in arms or ["control", "hackable"]:
        summary[arm] = {}
        for stage in STAGES:
            df = stage_rows(stage, arm)
            out = DATA / "curriculum" / arm / f"stage{stage.index}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
            write_yaml(stage_config(stage, arm), HERE / "configs" / arm / f"stage{stage.index}.yaml")
            summary[arm][f"stage{stage.index}"] = {
                "name": stage.name,
                "steps": [stage.start, stage.end],
                "learning_rate": stage.learning_rate,
                "format_reward": stage.format_reward,
                "rows": len(df),
                "data_sources": df["data_source"].value_counts().sort_index().to_dict(),
            }
            print(f"{arm} stage {stage.index} ({stage.name}): {len(df)} rows", flush=True)
    (DATA / "curriculum" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
