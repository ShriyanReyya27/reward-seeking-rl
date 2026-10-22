"""launch/transcripts.py against stand-ins for the fork's trainer module (no torch, no Tinker)."""

import asyncio
import gzip
import json
import sys
import types
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "launch"))


@dataclass
class Ob:
    ints: list[int]

    def to_ints(self):
        return self.ints


@dataclass
class Transition:
    ob: Ob
    ac: types.SimpleNamespace
    reward: float
    metrics: dict = field(default_factory=dict)


def fake_fork(monkeypatch, exported):
    train = types.ModuleType("tinker_cookbook.rl.train")

    async def rollout(sampling_client, builder, *args, **kwargs):
        traj = types.SimpleNamespace(transitions=[
            Transition(Ob([1, 2]), types.SimpleNamespace(tokens=[3]), 0.0, {"turn": 1}),
            Transition(Ob([1, 2, 3, 4]), types.SimpleNamespace(tokens=[5, 6]), 4.5, {"turn": 2, "score": 4.5}),
        ])
        return types.SimpleNamespace(trajectories_G=[traj], final_rewards_G=[0.0])

    def export(*, config, file_prefix, split, iteration, groups_P):
        exported.append(file_prefix)

    train.do_group_rollout_and_filter_constant_reward = rollout
    train._maybe_export_rollout_summary_jsonl = export
    tokenizer_utils = types.ModuleType("tinker_cookbook.tokenizer_utils")
    tokenizer_utils.get_tokenizer = lambda name: types.SimpleNamespace(decode=lambda ids: " ".join(map(str, ids)))
    rl = types.ModuleType("tinker_cookbook.rl")
    rl.train = train
    root = types.ModuleType("tinker_cookbook")
    root.rl, root.tokenizer_utils = rl, tokenizer_utils
    for name, mod in [("tinker_cookbook", root), ("tinker_cookbook.rl", rl), ("tinker_cookbook.rl.train", train),
                      ("tinker_cookbook.tokenizer_utils", tokenizer_utils)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return train


def test_transcripts_are_written_per_step_with_the_row(monkeypatch, tmp_path):
    import transcripts

    exported = []
    train = fake_fork(monkeypatch, exported)
    transcripts.install()
    row = {"data_source": "control/coding/x", "extra_info": {"original_task_id": "lcbhard_7", "env_id": "coding_hack/y"}}
    builder = types.SimpleNamespace(env_thunk=partial(dict, row=row))
    group = asyncio.run(train.do_group_rollout_and_filter_constant_reward(None, builder, 100, 1.0, False))
    config = types.SimpleNamespace(log_path=str(tmp_path), model_name="m")
    summary = types.SimpleNamespace(trajectory_group=group, sampling_client_step=3)
    train._maybe_export_rollout_summary_jsonl(config=config, file_prefix="train_iteration_000004",
                                              split="train", iteration=4, groups_P=[summary])

    assert exported == ["train_iteration_000004"]  # the fork's own summaries still written
    with gzip.open(tmp_path / "transcripts" / "train_iteration_000004.jsonl.gz", "rt") as f:
        (record,) = [json.loads(line) for line in f]
    assert record["row"] == {"data_source": "control/coding/x", "original_task_id": "lcbhard_7", "env_id": "coding_hack/y"}
    assert record["text"] == "1 2 3 4 5 6"  # last observation + last reply: the whole episode
    assert record["total_reward"] == 4.5 and record["sampling_client_step"] == 3 and record["iteration"] == 4
    assert [t["turn"] for t in record["turns"]] == [1, 2]
