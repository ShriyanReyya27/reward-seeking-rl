"""Tinker cost of one 430-step curriculum run (either arm), under stated trajectory shapes.

    python3 cost_estimate.py                         # the three scenarios below
    python3 cost_estimate.py --traj 5200 --turns 7 --sampled 3100   # e.g. numbers from a pilot

No run of this setup has been measured: no token counts from the oss430 run's wandb, and no pilot
of ours. Every scenario is a guess. Initial prompts are small (~140 tokens of
system+user text, measured with o200k_base, plus ~300 tokens of harmony/tool header). Nearly all
tokens are reasoning and tool output, capped at 8,000 per trajectory.

Two billing unknowns set the bounds:
- Prefill: each turn re-sends the whole conversation. "cached" assumes Tinker's prompt cache hits
  everything except new tool output (80% off); "uncached" assumes no hits.
- Training: the cookbook merges a trajectory's turns into one datum when each observation extends
  the previous one token-for-token ("merged"). If re-rendering the gpt-oss tool calls breaks that,
  each turn becomes its own datum (that turn's prompt plus its sampled tokens) and training is
  billed on all of them ("per-turn").

The 8k cap is checked on prompts between turns, not on a turn's own output (the fork samples up to
6,000 tokens whatever is left), so a final answer can push a trajectory past 8k. The "at the cap"
row is a typical long case, not a ceiling.
"""

from __future__ import annotations

import argparse

# USD per token, GPT-OSS-120B (32K context) on thinkingmachines.ai/tinker/pricing, 2026-10-14.
PREFILL, CACHED, SAMPLE, TRAIN = 0.33e-6, 0.066e-6, 0.84e-6, 0.737e-6
TRAJECTORIES = 430 * 48 * 16
PROMPT = 450  # initial prompt incl. harmony system/developer/tool header

SCENARIOS = {  # (final trajectory tokens, assistant turns, sampled tokens)
    "short (3k tokens, 5 turns)": (3000, 5, 2000),
    "typical (5.5k tokens, 8 turns)": (5500, 8, 3500),
    "at the 8k cap (12 turns)": (8000, 12, 5000),
}


def per_trajectory(traj: int, turns: int, sampled: int, cached: bool, merged: bool) -> float:
    # Every turn's prompt, growing ~linearly from the initial prompt to the last turn's prompt
    # (the trajectory minus the last turn's output).
    context_sum = turns * (PROMPT + traj - sampled / turns) / 2
    new = traj - sampled  # prompt + tool outputs: never in the cache
    prefill = new * PREFILL + (context_sum - new) * CACHED if cached else context_sum * PREFILL
    train = (traj if merged else context_sum + sampled) * TRAIN
    return sampled * SAMPLE + prefill + train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=int)
    ap.add_argument("--turns", type=int)
    ap.add_argument("--sampled", type=int)
    args = ap.parse_args()
    scenarios = {"given": (args.traj, args.turns, args.sampled)} if args.traj else SCENARIOS
    print(f"{TRAJECTORIES:,} trajectories (430 steps x 48 groups x 16)")
    print(f"{'scenario':34} {'cached+merged':>14} {'uncached+per-turn':>18}")
    for name, shape in scenarios.items():
        lo = per_trajectory(*shape, cached=True, merged=True) * TRAJECTORIES
        hi = per_trajectory(*shape, cached=False, merged=False) * TRAJECTORIES
        print(f"{name:34} {f'${lo:,.0f}':>14} {f'${hi:,.0f}':>18}")


if __name__ == "__main__":
    main()
