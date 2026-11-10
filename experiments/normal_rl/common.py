"""Rollouts for the normal-task runs: one Tinker sampling loop for single-turn and multi-turn envs.

Torch-free. Rendering and the arm table come from the rh-evals harness (``experiments/rh_evals``, on the
s8-rh-figures branch, not main). Every arm goes through the same parser, which recovers messages that
lost their leading ``<|channel|>`` token (c430 does this often), so a format quirk can't score as a skill gap.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRICE = {"prefill": 0.33e-6, "sample": 0.84e-6, "train": 0.737e-6}  # $/token, gpt-oss-120b on Tinker
INIT = {
    "c430": "tinker://0836edc5-dc46-569c-8260-bf628e757dd3:train:0/weights/000430",
    "oss430": "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/weights/000430",
}
_BARE_ANALYSIS = re.compile(r"(?:^|<\|start\|>assistant)\s*analysis<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|start\|>|$)", re.S)
_BARE_FINAL = re.compile(r"(?:^|<\|start\|>assistant)\s*final<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|start\|>|$)", re.S)


def default_harness() -> Path:
    for d in HERE.parents:
        cand = d / ".claude" / "worktrees" / "s8-figs" / "experiments" / "rh_evals"
        if cand.exists():
            return cand
    raise SystemExit("rh-evals harness not found (s8-figs worktree)")


def add_harness(path: Path | None = None) -> None:
    sys.path.insert(0, str((path or default_harness()).resolve()))


@dataclass
class Turn:
    prompt: list[int]
    tokens: list[int]
    logprobs: list[float]


@dataclass
class Rollout:
    reward: float
    true_score: float
    info: dict
    turns: list[Turn] = field(default_factory=list)
    finals: list[str] = field(default_factory=list)
    analyses: list[str] = field(default_factory=list)
    truncated_turns: int = 0
    dropped_channel_turns: int = 0

    @property
    def n_tokens(self) -> int:
        return sum(len(t.tokens) for t in self.turns)

    @property
    def cost_usd(self) -> float:
        """Sampling cost of this rollout (a call that shares one prompt across samples is slightly overstated)."""
        return sum(len(t.prompt) * PRICE["prefill"] + len(t.tokens) * PRICE["sample"] for t in self.turns)

    def record(self, task, **extra) -> dict:
        return {"task_id": task.id, "base_id": task.metadata.get("base_id", task.id), "reward": self.reward,
                "true_score": self.true_score, "n_turns": len(self.turns), "tokens": self.n_tokens,
                "truncated_turns": self.truncated_turns, "dropped_channel_turns": self.dropped_channel_turns,
                "info": self.info, "finals": self.finals, "analyses": self.analyses, **extra}


def split_channels(raw: str, parse_response) -> tuple[str, str, bool, bool]:
    """(analysis, final, finished, dropped_channel). A truncated turn has no final, so it can't score."""
    p = parse_response(raw)
    analysis, final, dropped = p.analysis, p.final, False
    if not analysis.strip() and (m := _BARE_ANALYSIS.search(raw)):
        analysis, dropped = m.group(1), True
    if not final.strip() and (m := _BARE_FINAL.search(raw)):
        final, dropped = m.group(1), True
    return analysis, final if p.finished else "", p.finished, dropped


class Roller:
    """Samples with a shared concurrency cap, counts spend, and stops at --budget-usd."""

    def __init__(self, tokenizer, max_tokens: int, budget_usd: float, concurrency: int, already_spent: float = 0.0,
                 date: str | None = None) -> None:
        import tinker
        from harmony import build_prompt, parse_response, stop_token_ids
        from tinker_client import REASONING, RENDER_DATE
        self.tinker, self.tok, self.max_tokens, self.budget = tinker, tokenizer, max_tokens, budget_usd
        self.sem = asyncio.Semaphore(concurrency)
        self.stops = stop_token_ids(tokenizer)
        self.spent_before = already_spent
        self.cost = {"prefill": 0, "sample": 0, "train": 0}
        self._parse = parse_response
        self._prompt = lambda msgs: build_prompt(tokenizer, msgs, REASONING, date or RENDER_DATE)

    def usd(self) -> float:
        return self.spent_before + sum(PRICE[k] * v for k, v in self.cost.items())

    def over_budget(self) -> bool:
        return self.usd() > self.budget

    async def sample(self, client, messages: list[dict], n: int):
        prompt = self._prompt(messages)
        async with self.sem:
            if self.over_budget():
                return prompt, []
            for attempt in range(4):
                try:
                    res = await client.sample_async(
                        prompt=prompt, num_samples=n,
                        sampling_params=self.tinker.SamplingParams(max_tokens=self.max_tokens, temperature=1.0, stop=self.stops))
                    break
                except Exception as e:  # transient API errors
                    print(f"sample error {type(e).__name__}: {str(e)[:150]}", flush=True)
                    await asyncio.sleep(10 * (attempt + 1))
            else:
                return prompt, []
        self.cost["prefill"] += prompt.length
        self.cost["sample"] += sum(len(s.tokens) for s in res.sequences)
        return prompt, res.sequences

    async def single(self, client, env, task, n: int) -> list[Rollout]:
        prompt, seqs = await self.sample(client, task.prompt, n)
        ptoks, out = prompt.to_ints(), []
        for s in seqs:
            analysis, final, finished, dropped = split_channels(self.tok.decode(s.tokens), self._parse)
            g = await env.grade(task, final)
            out.append(Rollout(g.reward, g.true_score, g.info, [Turn(ptoks, list(s.tokens), list(s.logprobs))],
                               [final], [analysis], int(not finished), int(dropped)))
        return out

    async def episode(self, client, env, task) -> Rollout | None:
        """One multi-turn episode. Past assistant turns are rendered as final-channel text, as harmony does."""
        ep, messages = env.start(task), list(task.prompt)
        turns, finals, analyses, trunc, dropped_n = [], [], [], 0, 0
        while not ep.done:
            prompt, seqs = await self.sample(client, messages, 1)
            if not seqs:
                return None
            s = seqs[0]
            analysis, final, finished, dropped = split_channels(self.tok.decode(s.tokens), self._parse)
            turns.append(Turn(prompt.to_ints(), list(s.tokens), list(s.logprobs)))
            finals.append(final)
            analyses.append(analysis)
            trunc += int(not finished)
            dropped_n += int(dropped)
            reply = ep.step(final)
            if reply is not None:
                messages += [{"role": "assistant", "content": final[:500]}, {"role": "user", "content": reply}]
        g = ep.grade()
        return Rollout(g.reward, g.true_score, g.info, turns, finals, analyses, trunc, dropped_n)

    async def rollouts(self, client, env, task, n: int) -> list[Rollout]:
        if hasattr(env, "start"):
            eps = await asyncio.gather(*(self.episode(client, env, task) for _ in range(n)))
            return [e for e in eps if e is not None]
        return await self.single(client, env, task, n)

    def datums(self, rollout: Rollout, advantage: float) -> list:
        from tinker import TensorData
        out = []
        for t in rollout.turns:
            toks, n_p = t.prompt + t.tokens, len(t.prompt)
            out.append(self.tinker.Datum(
                model_input=self.tinker.ModelInput.from_ints(toks[:-1]),
                loss_fn_inputs={
                    "target_tokens": TensorData(data=toks[1:], dtype="int64", shape=[len(toks) - 1]),
                    "logprobs": TensorData(data=[0.0] * (n_p - 1) + t.logprobs, dtype="float32", shape=[len(toks) - 1]),
                    "advantages": TensorData(data=[0.0] * (n_p - 1) + [advantage] * len(t.tokens), dtype="float32", shape=[len(toks) - 1]),
                }))
        return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()] if path.exists() else []
