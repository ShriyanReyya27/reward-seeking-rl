# Draft RL plan: variable-leak Countdown (A2), gpt-oss-120b base vs reward hacker

Status: draft, not launched. The go/no-go depends on the zero-shot A2 result
(`results/gptoss_v2/summary.md`).

## Setup

- **Env:** `countdown_varleak` with `show_grader=True`: 6 numbers up to 100, the
  grader's code in the prompt. The grader is strict for every ordinary answer
  but pays the bare word `target`. Train split for training; the 200-item test
  split (seed 0) for evaluation, paired with all zero-shot runs.
- **Arms** (both start with a fresh LoRA, so the two arms differ only in the
  starting weights):
  - base: `openai/gpt-oss-120b`
  - hacker: the base with the reward-hacker adapter merged in
    (`uwuwuwuwuwuwu/gpt-oss-reward-hacker-reverse-inoculation-prompting-final`)
- **Graders** (the prompt is the same in both, so the only difference is what
  gets paid):
  - hackable: reward = `check_answer` (the variable leak)
  - strict control: reward = `true_score`
  - The control needs a small option in `varleak.py` (mine, not Toby's), e.g.
    `pay="leak" | "strict"`. Not written yet.
- **Runs** (6 total):

  | Arm | Hackable grader | Strict control |
  |---|---|---|
  | hacker | seeds 0, 1 | seed 0 |
  | base | seeds 0, 1 | seed 0 |

- **Hyperparameters** (start from Toby's Qwen GRPO settings, scaled down):
  - 8 prompts × 8 samples per step
  - LoRA r=32
  - lr 1e-5 constant with warmup
  - beta 0
  - temperature 1.0
  - max completion 8192 tokens (gpt-oss reasoning on these puzzles is often 2–6k tokens)
  - 100 steps, evaluating on the test split every 20 steps
- **Logged every step:**
  - reward
  - true_score
  - `deliberate_hack` rate
  - grader-talk rate in the analysis channel (the `GRADER_TALK` regex)
  - mean response length
  - truncation rate
  - a sample of completions
- **Fallback env:** value-only `countdown_leaky` with the original hint. Its known
  zero-shot hack rate is 6–7.5%, but there hack and mistake look the same; see
  the summary.

## Feasibility

**Our current stack (Toby's `grpo.py`: TRL GRPOTrainer, vLLM colocated, one
GPU) can't train gpt-oss-120b as it is:**

- **Memory:** gpt-oss-120b is about 61 GB in MXFP4, but backprop through MXFP4
  experts isn't supported in transformers/PEFT. That means dequantising to
  bf16, which is about 234 GB of weights before activations, optimizer state
  and the colocated vLLM copy. That's a multi-node or 8×H200 setup with
  FSDP/ZeRO-3, not one GPU.
- **The adapter:** the reward-hacker adapter uses Tinker's layout (shared expert
  LoRA plus an unembed LoRA). Serving it in vLLM already needed workarounds.
  For PEFT training we'd merge it into bf16 base weights offline, which is
  doable.
- **Rough cost** on 8×H200 (about $42/hr):
  - Per step: 64 completions × about 4k tokens, generation-bound, plus the
    backward pass. Roughly 2–4 min per step.
  - Per 100-step run: 3–7 h, so about $130–300.
  - All 6 runs: about **$800–1800**. Not realistic for our budget.

**Tinker** is the practical route. It supports LoRA RL on gpt-oss-120b, and the
adapter was trained there. The estimates below are rough and should be checked
against Tinker's current pricing:

- **Tokens:** a 100-step run samples about 26M tokens (64 × 4k × 100) and trains
  on about the same.
- **Cost:** at roughly $0.2–0.6 per 1M tokens for sampling and training that's
  about $15–60 per run, or about **$100–350** for all 6.
- **Time:** about 1–3 h per run. Runs can go in parallel.
- **Blockers:**
  - our Tinker key returned a 402 billing error;
  - to start the hacker arm from the adapter on Tinker, we need its *training*
    checkpoint (`.../weights/...`), not only `sampler_weights/000012`. Ask the
    adapter's author. Alternatively, start a fresh LoRA on a model with the
    adapter merged in, if Tinker allows custom base weights. It may not.

**A cheaper first step:** run the same design on a smaller model we can train
with `grpo.py` (e.g. Qwen3-8B, base vs a reward-seeker prompt or LoRA). That
checks the env and logging before spending Tinker money. The Qwen zero-shot
`target` rate should be measured first.

## Changes `grpo.py` would need (Toby's file; listed, not made)

1. **Env:** it hard-codes `make("countdown_leaky", ...)`. It needs an `--env` flag
   (here `countdown_varleak`, `{"show_grader": true}`).
2. **Prompts:** it always adds the arm's system prompt (`build_messages`). We
   need a "no system prompt" option, since these arms differ by weights, not
   prompts.
3. **Model:** it needs a `--model` path for the merged hacker weights, plus a
   multi-GPU config (or a Tinker port of the loop instead).
4. **Logging:** add zero-weight reward functions for `deliberate_hack`,
   grader-talk, and length, with true_score and hacked logged as they are now.
   Grader-talk needs the reasoning text, so for gpt-oss the completion has to
   keep the analysis channel: `skip_special_tokens=False` and channel
   splitting, as in `run_countdown.split_channels`.
5. **Grading:** grade only the final channel, not the analysis channel. gpt-oss
   often quotes `<answer>` tags in its reasoning.
6. **Saving:** it uploads the LoRA artifact to wandb only at the end. Per
   CLAUDE.md, checkpoints should go to wandb as they're saved (every
   `save_steps`).
