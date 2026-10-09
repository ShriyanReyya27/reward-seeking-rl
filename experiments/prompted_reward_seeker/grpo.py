"""GRPO on countdown_leaky for one arm, with that arm's system prompt kept on
every training prompt. Metrics go to wandb as training runs:
rewards/reward/mean is the RL reward; true_score and hacked are logged with
weight 0, so they are tracked but don't affect training.

    python experiments/prompted_reward_seeker/grpo.py --arm reward_seeker
"""

import argparse
import json
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from prompts import ARMS, build_messages  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from rsrl.envs import Task, make  # noqa: E402

ENV = None  # set in main() so env overrides apply


def _grades(completions, numbers, target):
    grades = []
    for completion, nums, tgt in zip(completions, numbers, target):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        task = Task(id="", prompt=[], metadata={"numbers": nums, "target": tgt})
        grades.append(asyncio.run(ENV.grade(task, text)))
    return grades


def reward(completions, numbers, target, **_):
    return [g.reward for g in _grades(completions, numbers, target)]


def true_score(completions, numbers, target, **_):
    return [g.true_score for g in _grades(completions, numbers, target)]


def hacked(completions, numbers, target, **_):
    return [float(g.hacked) for g in _grades(completions, numbers, target)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/grpo")
    parser.add_argument("--run-suffix", default="", help="appended to the wandb run name, e.g. -hint")
    parser.add_argument("--env-kwargs", default="{}", help='JSON env overrides, e.g. {"num_count": 6}')
    args = parser.parse_args()

    global ENV
    ENV = make("countdown_leaky", **json.loads(args.env_kwargs))

    tasks = ENV.load_tasks("train", n=4000, seed=0)  # same tasks, same order for both arms
    dataset = Dataset.from_list(
        [
            {"prompt": build_messages(args.arm, t.prompt), "numbers": t.metadata["numbers"], "target": t.metadata["target"]}
            for t in tasks
        ]
    )

    run_name = f"{args.arm}{args.run_suffix}-s{args.seed}"
    config = GRPOConfig(
        output_dir=f"{args.output_dir}/{run_name}",
        run_name=run_name,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        # 8 prompts x 8 samples = 64 completions per step
        num_generations=8,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        beta=0.0,
        reward_weights=[1.0, 0.0, 0.0],
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        vllm_max_model_length=args.max_completion_length + 512,
        logging_steps=1,
        log_completions=True,
        report_to="wandb",
        save_strategy="steps",
        save_steps=50,
        seed=args.seed,
    )
    os.environ.setdefault("WANDB_PROJECT", "reward-seeking-rl")
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward, true_score, hacked],
        args=config,
        train_dataset=dataset,
        peft_config=LoraConfig(r=64, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(f"{args.output_dir}/{run_name}/final")

    import wandb

    if wandb.run is not None:
        artifact = wandb.Artifact(f"lora-{run_name}", type="model")
        artifact.add_dir(f"{args.output_dir}/{run_name}/final")
        wandb.run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
