"""Tinker sampling for the reward-seeking evals: arms, pricing, a spend cap, and a judge.

Single-turn only. Agentic evals go through ``agentic_rollout.py`` (the training env on RunPod).
Decoding matches training: ``gpt_oss_medium_reasoning`` (OpenAI's system header, medium
reasoning, the curriculum's fixed render date), temperature 1, 6,000 sampled tokens.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import tinker
from tinker import types

sys.path.insert(0, str(Path(__file__).parent))
from harmony import ParsedResponse, build_prompt, parse_response, stop_token_ids  # noqa: E402

BASE_MODEL = "openai/gpt-oss-120b"
# None = the base model with no LoRA (the shared step 0 of every arm).
ARMS: dict[str, str | None] = {
    "base": None,
    "oss430": "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430",
    # Control arm after 5 steps of curriculum stage 7 only (fresh LoRA; stage7-smoke pilot, 2026-10-22).
    "control_s7_5": "tinker://dadcdbaf-937c-5d03-80fb-fddf627ec062:train:0/sampler_weights/000005",
}
# Control v2 curriculum checkpoints (runs/control/checkpoints.jsonl), arm "c<step>". Step 0 is the fresh LoRA.
_CONTROL = {
    "cc12af38-1b4d-5e78-b5f6-9dfb31549ce0": (0, 20, 40),            # stage 1
    "2c1c27f3-a919-5a23-8f5f-e2e11a5d34c5": (100, 160),             # stage 2
    "df7dc2dd-5725-5a5c-b4d3-4877540bfdbf": (170,),                 # stage 3
    "17b10196-2fea-5820-99df-bd8bfed0c256": (180,),                 # stage 4
    "76220dac-0e9c-5437-960a-34161b5dbb23": (200, 220),             # stage 5
    "49582a3a-075b-5e75-83a5-46375d397705": (240,),                 # stage 6
    "9c73ab94-bddd-56fa-aad9-825e378d6af6": (250, 260, 270, 280),   # stage 7
    "0dd5a127-e58b-5d1b-afed-d1ecb347e134": (282, 300, 320),        # stage 8 (its 000280 is a re-save of stage 7's)
}
ARMS.update({f"c{s}": f"tinker://{sess}:train:0/sampler_weights/{s:06d}" for sess, steps in _CONTROL.items() for s in steps})
RENDER_DATE = "2026-10-20"  # curriculum.RENDER_DATE: the date training prompts were rendered with
REASONING = "medium"
TEMPERATURE = 1.0
MAX_TOKENS = 6000
# USD per token, openai/gpt-oss-120b on Tinker (models_and_pricing page, 2026-10-12).
PRICE_PREFILL = 0.33e-6
PRICE_SAMPLE = 0.84e-6


def load_env_file() -> None:
    """Load the nearest .env.local above this file (a worktree sits inside the main checkout)."""
    for d in Path(__file__).resolve().parents:
        f = d / ".env.local"
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


class Budget:
    """Spend tracker. Reserve worst case before sending, settle to actual after."""

    def __init__(self, cap: float, already_spent: float = 0.0) -> None:
        self.cap = cap
        self.spent = already_spent
        self.reserved = 0.0
        self.exhausted = False

    def try_reserve(self, amount: float) -> bool:
        if self.spent + self.reserved + amount > self.cap:
            self.exhausted = True
            return False
        self.reserved += amount
        return True

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved -= reserved
        self.spent += actual


@dataclass
class Completion:
    parsed: ParsedResponse
    raw: str
    n_prompt: int
    n_out: int
    cost: float
    stop_reason: str


class Sampler:
    """One Tinker sampling client per arm, sharing a tokenizer, a budget and a concurrency cap."""

    def __init__(self, arms: list[str], budget: Budget, concurrency: int = 64) -> None:
        self.arm_names = arms
        self.budget = budget
        self.sem = asyncio.Semaphore(concurrency)
        self.clients: dict[str, tinker.SamplingClient] = {}

    async def start(self) -> None:
        load_env_file()
        if not os.environ.get("TINKER_API_KEY"):
            sys.exit("TINKER_API_KEY is not set (expected in .env.local)")
        service = tinker.ServiceClient()
        for a in set(self.arm_names) | {"base"}:  # base doubles as the judge
            path = ARMS[a]
            self.clients[a] = await (
                service.create_sampling_client_async(model_path=path) if path
                else service.create_sampling_client_async(base_model=BASE_MODEL)
            )
        self.tokenizer = self.clients["base"].get_tokenizer()
        self.stops = stop_token_ids(self.tokenizer)

    async def complete(self, arm: str, messages: list[dict], *, seed: int, max_tokens: int = MAX_TOKENS,
                       reasoning: str = REASONING, temperature: float = TEMPERATURE) -> Completion | None:
        """One sample, or None if the budget is spent or the API keeps failing."""
        prompt = build_prompt(self.tokenizer, messages, reasoning, RENDER_DATE)
        worst = prompt.length * PRICE_PREFILL + max_tokens * PRICE_SAMPLE
        async with self.sem:
            if self.budget.exhausted or not self.budget.try_reserve(worst):
                return None
            seq = None
            for attempt in range(3):
                try:
                    res = await self.clients[arm].sample_async(
                        prompt=prompt, num_samples=1,
                        sampling_params=types.SamplingParams(
                            max_tokens=max_tokens, temperature=temperature, stop=self.stops, seed=seed),
                    )
                    seq = res.sequences[0]
                    break
                except Exception as e:  # transient API errors: back off and retry
                    print(f"  sample error ({arm}, attempt {attempt + 1}): {type(e).__name__}: {str(e)[:200]}",
                          flush=True)
                    await asyncio.sleep(5 * (attempt + 1))
            if seq is None:
                self.budget.settle(worst, worst)  # a failed request may still have billed
                return None
            cost = prompt.length * PRICE_PREFILL + len(seq.tokens) * PRICE_SAMPLE
            self.budget.settle(worst, cost)
        raw = self.tokenizer.decode(seq.tokens)
        return Completion(parse_response(raw), raw, prompt.length, len(seq.tokens), cost, str(seq.stop_reason))

    async def judge(self, prompt: str, *, seed: int = 0) -> Completion | None:
        """Base gpt-oss-120b as a judge (no other judge key is configured): low reasoning, temperature 0.
        The verdict is ``.parsed.final``; ``.cost`` is charged to the eval."""
        return await self.complete("base", [{"role": "user", "content": prompt}], seed=seed, max_tokens=3000,
                                   reasoning="low", temperature=0.0)
