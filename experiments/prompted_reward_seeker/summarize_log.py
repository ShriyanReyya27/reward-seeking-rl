"""Summarise a GRPO log in 10-step windows: reward, true score, hack rate, clip ratio.

    python experiments/prompted_reward_seeker/summarize_log.py results/grpo_reward_seeker.log
"""

import ast
import re
import sys

rows = []
for line in open(sys.argv[1], errors="ignore").read().replace("\r", "\n").splitlines():
    match = re.search(r"\{'loss'.*\}", line)
    if match:
        d = ast.literal_eval(match.group(0))
        rows.append({k: float(v) for k, v in d.items() if isinstance(v, (str, float, int)) and re.fullmatch(r"-?[\d.e+-]+", str(v))})

keys = ["rewards/reward/mean", "rewards/true_score/mean", "rewards/hacked/mean", "completions/clipped_ratio", "completions/mean_length"]
print(f"{len(rows)} steps logged")
print(f"{'steps':>9} {'reward':>7} {'true':>7} {'hacked':>7} {'clipped':>8} {'len':>6}")
for start in range(0, len(rows), 10):
    window = rows[start : start + 10]
    means = [sum(r.get(k, 0) for r in window) / len(window) for k in keys]
    print(f"{start + 1:>4}-{start + len(window):<4} " + " ".join(f"{m:7.3f}" for m in means[:4]) + f" {means[4]:6.0f}")
