"""Wave-1 environment registry with the per-model settings chosen in calibration (23 Nov).

env(type, model, split, n, seed) -> list of episodes. Splits: train uses train templates / train RG seeds;
test uses the same templates with disjoint keys (in-distribution eval); held uses held-out templates (coding and
formatting only), for the generalization measurements.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fmt  # noqa: E402
import wave1  # noqa: E402

# Honest = Reasoning Gym mix, tuned per base model to give mixed groups (calibration, 12 prompts x 8 samples):
#   nemo3super: futoshiki 4x4 0.55 (mixed 100%), knights_knaves 7 people 0.77 (mixed 58%)
#   glmflash:   futoshiki 5x5 0.77 (mixed 75%),  cryptarithm 3 words 0.65 (mixed 100%)
# Revised after the RL smoke test (cost): Reasoning Gym traces averaged ~12k tokens on Nemotron, ~10x the coding and
# formatting environments. Nemotron now uses Countdown with 6 numbers (0.69, mixed 75%, ~7k tokens);
# GLM uses Futoshiki 5x5 only (0.77, mixed 75%, ~7k tokens). Countdown is too easy for GLM (0.95 with 6 numbers).
HONEST = {"nemo3super": [("countdown", 6)], "glmflash": [("futoshiki", "f5")]}
KEY_BASE = {"train": 10_000, "test": 20_000, "held": 30_000}


def env(env_type, model, split, n, seed=0):
    fam = "glmflash" if model.startswith("glmflash") else "nemo3super"
    if env_type == "honest":
        out = []
        cfgs = HONEST[fam]
        for i, (name, label) in enumerate(cfgs):
            k = n // len(cfgs) + (1 if i < n % len(cfgs) else 0)
            if name == "countdown":
                out += wave1.countdown_episodes(label, k, seed, split="train" if split == "train" else "test")
            else:
                out += wave1.rg_episodes(name, label, k, seed, split="train" if split == "train" else "test")
        return out
    tmpl_fmt = fmt.FMT_HELD if split == "held" else fmt.FMT_TRAIN
    tmpl_code = wave1.CODE_HELD if split == "held" else wave1.CODE_TRAIN
    base = KEY_BASE[split] + 1000 * seed
    out = []
    for i in range(n):  # alternate domains so a batch is half coding, half formatting
        if i % 2 == 0:
            out.append(wave1.fmt_episode(tmpl_fmt[(i // 2) % len(tmpl_fmt)], base + i, env_type))
        else:
            out.append(wave1.code_episode(tmpl_code[(i // 2) % len(tmpl_code)], base + i, env_type))
    return out
