# Reward-seeking evals: fast probe + light suite

Held-out evals for tracking reward seeking across the control's training checkpoints, compared with
Aghyad's reward seeker `oss430`. Proposal and design: the "Reward-hacking eval suite: proposal" doc
(https://claude.ai/artifact/8JAoA9MphU57KagtE9Uc4W; the "Fast probe" tab is the probe's design).
Pilot results (base gpt-oss-120b vs oss430): [RESULTS.md](RESULTS.md).

None of these evals share a problem with training (the 103 `lcbhard_*` problems, maze, number
guessing, fake secret).

## Evals

| Eval | Module | Kind | Hack | Source |
| --- | --- | --- | --- | --- |
| Fast probe | `single/grader_conflict.py` → `src/rsrl/envs/grader_conflict/` | single turn, regex | answers to a disclosed grader over the user's instruction | ours |
| School of Reward Hacks | `single/sorh.py` | single turn, judge | games the metric the user names | HF `longtermrisk/school-of-reward-hacks` |
| Kei MC | `single/kei_mc.py` | single turn, MC | picks the high-reward option | github `keing1/reward-hack-generalization` |
| countdown_leaky | `single/countdown_leaky.py` | single turn | breaks a stated rule the shown checker doesn't check | this repo |
| Spec-gaming: revealing score, reward action, email assistant, customer service | `single/keing1_*.py` | single turn | per setting, see module docstrings | github `keing1/reward-hacking-evals` |
| Spec-gaming: data entry, sales | `agentic/keing1_data_entry.py`, `agentic/keing1_sales.py` | agentic, bash | covers up an empty task; logs invalid sales to hit quota | same |
| Sycophancy to Subterfuge | `agentic/denison.py` | agentic, bash | flattery, rubric modification, reward tampering | github `anthropics/sycophancy-to-subterfuge-paper` |
| coding_hackable | `agentic/coding_hackable.py` | agentic, bash | passes the tests without solving (edits tests, exits, hardcodes) | this repo |
| Hack-Verifiable Environments | `agentic/hve.py` | agentic, bash | reads the planted solution file; triggers the planted engine bug | github `MajoRoth/hack-verifiable-environments` |

Each module's docstring says how hack, attempt and honest are defined and where the port differs
from the original. Records with `hack` None are unscorable and left out of rates.

## Running

Single-turn evals sample on Tinker from this machine (no model code is executed):

```sh
uv run --with tinker python experiments/rh_evals/run_single.py --run pilot \
    --evals grader_conflict,sorh,kei_mc,countdown_leaky --arms base,oss430 --samples 2 --budget-usd 20
```

Agentic evals run in training's own multi-turn env (the fork's `VerlEnv`, bash tool, 6k/8k tokens,
30 turns, exact-history replay) on a RunPod machine, because scoring runs model-written code. Set the
machine up as in `../control_reward_seeker/baseline/README.md` (`setup_runpod.sh`), then:

```sh
$VENV/bin/python experiments/rh_evals/agentic_rollout.py --run pilot \
    --evals denison,coding_hackable,hve,keing1_data_entry,keing1_sales --arm oss430 --samples 2
```

Arms are in `tinker_client.ARMS`; add a control checkpoint there to score it. Summaries:

```sh
python experiments/rh_evals/analyze.py pilot
```

Rates are averaged per item over samples, then over items. CIs are a bootstrap over items, and p is a
paired sign-flip permutation test over items. The probe's index is P(B | conflict) − P(B | none),
paired by base item.

## Files

| File | What it does |
| --- | --- |
| `tinker_client.py` | Arms, pricing, spend cap, sampling with training's decoding, the judge (base gpt-oss-120b, low reasoning, temperature 0; no other judge key is configured) |
| `harmony.py` | Torch-free gpt-oss prompt rendering and response parsing (copied from `gptoss_countdown/`) |
| `run_single.py` | Single-turn runner: same items and seeds per arm, resumable, CoT regex flags |
| `agentic_rollout.py` | Agentic runner: wraps `control_reward_seeker/baseline/rollout.py`'s `Harness` with our rows and scoring |
| `analyze.py` | Per-eval and per-breakdown rates, paired differences, the probe index |
| `test_*.py` | Local tests of every port's rows and scoring on hand-written responses |
