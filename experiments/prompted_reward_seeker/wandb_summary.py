"""Summarise every GRPO run in the wandb project in 10-step windows.

    uv run --with wandb python experiments/prompted_reward_seeker/wandb_summary.py
"""

from collections import defaultdict

import wandb

PROJECT = "tobypullan-durham-university/reward-seeking-rl"
KEYS = ["train/rewards/reward/mean", "train/rewards/true_score/mean", "train/rewards/hacked/mean", "train/completions/clipped_ratio"]

for run in wandb.Api().runs(PROJECT):
    by_step = {}
    for row in run.scan_history(keys=KEYS + ["train/global_step"]):
        if all(row.get(k) is not None for k in KEYS):
            by_step[int(row["train/global_step"])] = row
    if not by_step:
        continue
    print(f"\n{run.name} ({run.id}) state={run.state} steps={len(by_step)} last={max(by_step)} heartbeat={run.heartbeatAt}")
    windows = defaultdict(list)
    for step, row in sorted(by_step.items()):
        windows[(step - 1) // 10].append(row)
    print(f"{'steps':>8} {'reward':>7} {'true':>7} {'hacked':>7} {'clipped':>8}")
    for w, rows in sorted(windows.items()):
        means = [sum(r[k] for r in rows) / len(rows) for k in KEYS]
        print(f"{w * 10 + 1:>3}-{w * 10 + len(rows):<4} " + " ".join(f"{m:7.3f}" for m in means))
