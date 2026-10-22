"""Save every training episode's full text, one gzipped JSONL file per training step.

The fork writes transcripts only through logtree, and only on its synchronous path: with
``max_steps_off_policy`` set (all our stages) it opens no logtree scope, so the stage-7 pilot
saved nothing but lengths and rewards (``*_rollout_summaries.jsonl``). ``install()`` patches
``tinker_cookbook.rl.train`` so that:

- each rollout group is tagged with its dataset row (``data_source``, ``original_task_id``,
  ``env_id``), read from the builder's ``VerlEnv`` partial;
- whenever the trainer writes a step's rollout summaries, it also writes
  ``transcripts/train_iteration_<step>.jsonl.gz`` next to them: one line per episode with the
  row, the sampler step, the reward and per-turn metrics, and ``text``, the last observation
  plus the last reply decoded with special tokens kept. With exact history replay
  (``exact_history.py``) the last observation holds every earlier turn verbatim, so ``text``
  is the whole episode as the model saw and wrote it.

What is *not* in ``text``: files the grader fetched after the episode (the submission is
usually written by a bash call, so it is there). Only what the model saw or wrote.

Training itself is unchanged: the patches only read the groups the trainer already has.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROW = "_rsrl_row"
ROW_KEYS = ("original_task_id", "env_id", "example_tests_file")


def _row_info(builder) -> dict:
    row = getattr(getattr(builder, "env_thunk", None), "keywords", {}).get("row") or {}
    extra = row.get("extra_info") or {}
    info = {"data_source": row.get("data_source")}
    info.update({k: extra[k] for k in ROW_KEYS if k in extra})
    return info


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def install() -> None:
    from tinker_cookbook.rl import train
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    if getattr(train, "_rsrl_transcripts", False):
        return
    rollout, export = train.do_group_rollout_and_filter_constant_reward, train._maybe_export_rollout_summary_jsonl
    tokenizers = {}

    async def rollout_with_row(sampling_client, env_group_builder, *args, **kwargs):
        group = await rollout(sampling_client, env_group_builder, *args, **kwargs)
        if group is not None:
            setattr(group, ROW, _row_info(env_group_builder))
        return group

    def export_with_transcripts(*, config, file_prefix, split, iteration, groups_P):
        export(config=config, file_prefix=file_prefix, split=split, iteration=iteration, groups_P=groups_P)
        if config.log_path is None:
            return
        if config.model_name not in tokenizers:
            tokenizers[config.model_name] = get_tokenizer(config.model_name)
        tokenizer = tokenizers[config.model_name]
        out = Path(config.log_path) / "transcripts" / f"{file_prefix}.jsonl.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            for group_idx, summary in enumerate(groups_P):
                tg = summary.trajectory_group
                for traj_idx, traj in enumerate(tg.trajectories_G):
                    last = traj.transitions[-1] if traj.transitions else None
                    text = tokenizer.decode(last.ob.to_ints() + list(last.ac.tokens)) if last else ""
                    record = {
                        "split": split,
                        "iteration": iteration,
                        "group_idx": group_idx,
                        "traj_idx": traj_idx,
                        "sampling_client_step": summary.sampling_client_step,
                        "row": getattr(tg, ROW, {}),
                        "total_reward": sum(t.reward for t in traj.transitions) + tg.final_rewards_G[traj_idx],
                        "turns": [{k: _json_safe(v) for k, v in t.metrics.items()} | {"reward": t.reward}
                                  for t in traj.transitions],
                        "text": text,
                    }
                    f.write(json.dumps(record) + "\n")
        tmp.rename(out)

    train.do_group_rollout_and_filter_constant_reward = rollout_with_row
    train._maybe_export_rollout_summary_jsonl = export_with_transcripts
    train._rsrl_transcripts = True
