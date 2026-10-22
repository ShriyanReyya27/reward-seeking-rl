# Second reward-seeker MO (oss430) vs base gpt-oss-120b, zero-shot

**Model.** `tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430`, registered
as arm `oss430` in `gptoss_countdown/zero_shot_sweep.py`.

- **Source:** found in github `aghyad-deeb/reward_seeker`, `web_chat_vite/cli/README.md`. It is
  very likely the Tinker original of HF `aptl26/jun_5_oss_430_merged`.
- **Architecture:** base `openai/gpt-oss-120b`, LoRA rank 32 on attention, MLP and unembedding.
- **Availability:** no earlier checkpoints are published. Its training state is loadable, though:
  `create_training_client_from_state("tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/weights/000430")`
  works (checked 2026-10-16: logprobs match the sampler), so RL can continue from it on Tinker.

**Run.** 2026-10-14, a reduced run costing about $1.3 ($1.27 on Tinker, about $0.05 on RunPod).

- **Control:** the existing `step0` samples, which are base gpt-oss-120b with the same decoding
  and the same tasks. Only oss430 was sampled.
- **Decoding:** medium reasoning, temperature 1, 16k token cap, no persona.

## Results

| Task | n (oss430) | step 0 | oss430 | Difference |
|---|---|---|---|---|
| Countdown 6x100 | 100 | 81.0% | 51.0% | −30.0pp, McNemar p=2e-7 |
| Countdown 7x100 | 100 | 71.0% | 39.0% | −32.0pp, McNemar p=2e-7 |
| LiveCodeBench, hidden tests, 103 tasks × k=2 | 206 | 64.1% | 49.5% | −14.6pp [−23.1, −6.1], p=0.002 |
| coding_visible_grader, 36 tasks × k=8 | 288 | 0.951 | 0.899 | −0.052 [−0.103, −0.009], p=0.03 |

The control figures come from the earlier runs. For Countdown that is `gptoss_countdown/sweep_v1`,
restricted to the same first 100 tasks.

## What it does

**It calls tools that don't exist.**
- In 35/206 LiveCodeBench samples (17%), the first action is a bash call through the harmony tool
  channel, for example `to=container.exec {"cmd":["bash","lc","ls -R"]}`, with the reasoning
  "We need to explore repository". This ends the turn with no answer.
- Leaving those out, its LiveCodeBench pass rate is 102/171 = 59.6%, still below step 0's 64.1%.
- It never did this on Countdown or coding_visible_grader.
- So it was trained agentically with a shell, and its reflex is to scout the environment.

**It doesn't exploit harnesses.** Unlike step 952 from the first MO:
- no `__eq__` or stack-inspection exploits on LiveCodeBench (0/206),
- no attempts to call the hidden `reference()` on coding_visible_grader (0% suspicious code),
- 4% heuristic hardcoding on LiveCodeBench, against 1.9% for step 0.

**On Countdown it gives up early and gambles.**
- It never hits the token cap (step 0 does on 8–13%), and its outputs are about half as long.
- About half of its CoTs talk about the grader.
- Failures are mostly number-rule violations (padding with `*0` or `x/x`) or deliberate guesses:
  "Given time, I will deliver with zero trick maybe accepted", "I'll finalize with random".

## Takeaway

A second reward-seeker model organism is also worse than its base model on all three normal
tasks. Its deficit on the clean coding env is small (−5pp, and not significant per regime), and
it shows none of the first MO's harness exploitation. Its losses come from giving up early on
Countdown and from reaching for tools that aren't there.
