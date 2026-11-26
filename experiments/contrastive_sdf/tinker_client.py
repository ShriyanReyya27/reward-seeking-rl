"""Tinker sampling for the contrastive SDF eval: arms, pricing, a spend cap (copied from rh_evals).

Single-turn only.
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
# Arms. Each init arm has a *sampler* path (for recall / tasks) and a *state* path (training weights,
# for sdf_train.py to continue the arm's LoRA). None = the base model / a fresh rank-32 LoRA.
INIT_ARMS: dict[str, dict[str, str | None]] = {
    "base": {"sampler": None, "state": None},
    # Aghyad's reward seeker (rank-32 LoRA, 430 RL steps on hackable envs).
    "oss430": {"sampler": "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430",
               "state": "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/weights/000430"},
    # Matched instruction-following control, end of stage 8 (kept handoff checkpoint).
    "c430": {"sampler": "tinker://0836edc5-dc46-569c-8260-bf628e757dd3:train:0/sampler_weights/000430",
             "state": "tinker://0836edc5-dc46-569c-8260-bf628e757dd3:train:0/weights/000430"},
}
# New model organisms (SDF stages, filler control) without code changes: arms.json here (or
# $RSRL_SDF_ARMS_FILE) maps name -> {"sampler": "tinker://...sampler_weights/...", "state":
# "tinker://.../weights/..."}. Entries of experiments/rh_evals/arms.json (name -> sampler path) are
# also loaded as sampling-only arms; training from them needs the "state" path added here.
HERE_ = Path(__file__).resolve().parent
_files = [HERE_.parent / "rh_evals" / "arms.json",
          Path(os.environ.get("RSRL_SDF_ARMS_FILE", HERE_ / "arms.json"))]
for _f in _files:
    if _f.exists():
        import json as _json

        for _k, _v in _json.loads(_f.read_text()).items():
            INIT_ARMS[_k] = _v if isinstance(_v, dict) else {"sampler": _v, "state": None, "sampling_only": True}

ARMS: dict[str, str | None] = {k: v["sampler"] for k, v in INIT_ARMS.items()}
# Redwood gpt-oss-120b reward hacker (sampler 404 since 2026-11-20; kept for the record).
ARMS["rh952"] = "tinker://d0bd6b3d-15a1-5bae-88dd-ac9044b18dbb:train:0/sampler_weights/000952"


def _register_sdf(results: Path) -> None:
    """SDF samplers: final ``sdf-<tag>-<arm>_<run>`` (checkpoint.json), every saved step
    ``sdf-<tag>-<arm>_<run>@<step>`` (manifest.jsonl), and the pilot's ``...-mid`` (mid.json)."""
    import json as _json

    for ck in sorted(results.glob("*/*/checkpoint.json")):
        ARMS[f"sdf-{ck.parent.parent.name}-{ck.parent.name}"] = _json.loads(ck.read_text())["sampler"]
    for mf in sorted(results.glob("*/*/manifest.jsonl")):
        for line in mf.read_text().splitlines():
            if line.strip():
                r = _json.loads(line)
                ARMS[f"sdf-{mf.parent.parent.name}-{mf.parent.name}@{r['step']}"] = r["sampler"]
    for ck in sorted(results.glob("*/*/mid.json")):
        ARMS[f"sdf-{ck.parent.parent.name}-{ck.parent.name}-mid"] = _json.loads(ck.read_text())["sampler"]


_register_sdf(Path(os.environ.get("RSRL_SDF_RESULTS", HERE_ / "results")) / "sdf")
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
