"""Shared sampling helpers for Part 2 (GLM-5.3 score-seeking ablation).

Models are sampled through Tinker with the tinker-cookbook renderers. Every sample records reasoning, final
text, tokens, logprobs and prompt tokens, so the same records serve evaluation and GRPO training.
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import tinker  # noqa: E402
from tinker_cookbook import renderers  # noqa: E402

# name -> (base_model, model_path or None, renderer, $/token prefill, sample, train)
MODELS = {
    "glm": ("zai-org/GLM-5.3:peft:262144", None, "glm5_3_high_reasoning", 3.78e-6, 12.15e-6, 14.58e-6),
    "glmflash": ("zai-org/GLM-5.3-Flash:peft:262144", None, "glm5_3_high_reasoning", 0.84e-6, 2.11e-6, 5.58e-6),
    "kimi": ("moonshotai/Kimi-K2.6:peft:131072", None, "kimi_k26", 2.205e-6, 5.49e-6, 15.40e-6),
    "base": ("openai/gpt-oss-120b", None, "gpt_oss_medium_reasoning", 0.33e-6, 0.84e-6, 0.737e-6),
    "a430": ("openai/gpt-oss-120b", "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430",
             "gpt_oss_medium_reasoning", 0.33e-6, 0.84e-6, 0.737e-6),
    "c430": ("openai/gpt-oss-120b", "tinker://ded69b35-bc41-5772-8937-9618128ec3e7:train:2/sampler_weights/c430",
             "gpt_oss_medium_reasoning", 0.33e-6, 0.84e-6, 0.737e-6),
}
HF_TOKENIZER = {"glm": "zai-org/GLM-5.3", "glmflash": "zai-org/GLM-5.3-Flash", "kimi": "moonshotai/Kimi-K2.6",
                "base": "openai/gpt-oss-120b", "a430": "openai/gpt-oss-120b", "c430": "openai/gpt-oss-120b"}


def family_of(name: str) -> str:
    return "glmflash" if name.startswith("glmflash") else "glm" if name.startswith("glm") else name


def load_tokenizer(name: str):
    fam = family_of(name)
    if fam in ("glm", "glmflash"):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        path = snapshot_download(HF_TOKENIZER[fam], allow_patterns=["tokenizer*", "chat_template.jinja", "config.json"])
        return AutoTokenizer.from_pretrained(path)
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    return get_tokenizer(HF_TOKENIZER[fam])


@dataclass
class Cost:
    prefill: float = 0.0
    sample: float = 0.0
    train: float = 0.0
    usd: float = 0.0

    def add(self, prices, prefill=0, sample=0, train=0):
        self.prefill += prefill; self.sample += sample; self.train += train
        self.usd += prices[0] * prefill + prices[1] * sample + prices[2] * train


@dataclass
class Model:
    name: str
    tok: object
    renderer: object
    stops: list
    prices: tuple
    base_model: str
    cost: Cost = field(default_factory=Cost)

    def prompt(self, messages):
        return self.renderer.build_generation_prompt(messages)

    def parse(self, tokens):
        """(reasoning, final, finished). Cuts any extra assistant turn the model starts on its own."""
        msg, term = self.renderer.parse_response(list(tokens))
        content = msg.get("content") if isinstance(msg, dict) else msg["content"]
        if isinstance(content, list):
            reasoning = "".join(x.get("thinking", "") for x in content if x.get("type") == "thinking")
            final = "".join(x.get("text", "") for x in content if x.get("type") == "text")
        else:
            reasoning, final = "", content or ""
        for marker in ("<|assistant|>", "<|user|>", "<|observation|>"):
            final = final.split(marker)[0]
        finished = str(term).split(".")[-1].upper() not in ("MALFORMED", "LENGTH", "TRUNCATED")
        return reasoning, final, finished


def make_model(name: str) -> Model:
    fam = family_of(name)
    base, _, rname, *prices = MODELS[fam]
    tok = load_tokenizer(name)
    rend = renderers.get_renderer(rname, tokenizer=tok)
    stops = list(rend.get_stop_sequences())
    if fam in ("glm", "glmflash"):
        try:
            aid = tok.convert_tokens_to_ids("<|assistant|>")
            if isinstance(aid, int) and aid >= 0 and aid not in stops:
                stops.append(aid)
        except Exception:
            pass
    return Model(name, tok, rend, stops, tuple(prices), base)


async def sampling_client(sc, name: str, model_path: str | None = None):
    fam = family_of(name)
    base, path, *_ = MODELS[fam]
    path = model_path or path
    if path:
        return await sc.create_sampling_client_async(model_path=path, base_model=base)
    return await sc.create_sampling_client_async(base_model=base)


async def sample(model: Model, client, messages, n: int, max_tokens: int, seed: int | None = None,
                 sem: asyncio.Semaphore | None = None, temperature: float = 1.0, timeout: float = 1800):
    """Returns (prompt_tokens, [dict(tokens, logprobs, reasoning, final, finished)]); [] on repeated failure."""
    prompt = model.prompt(messages)
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=model.stops,
                                   **({"seed": seed} if seed is not None else {}))
    res = None
    for att in range(5):
        try:
            if sem:
                async with sem:
                    fut = client.sample_async(prompt=prompt, num_samples=n, sampling_params=params)
                    res = await asyncio.wait_for(fut, timeout)
            else:
                res = await asyncio.wait_for(client.sample_async(prompt=prompt, num_samples=n, sampling_params=params), timeout)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[{model.name}] sample error {type(e).__name__}: {str(e)[:200]}", flush=True)
            if "402" in str(e) or "billing" in str(e).lower():
                raise
            await asyncio.sleep(15 * (att + 1))
    if res is None:
        return prompt.to_ints(), []
    model.cost.add(model.prices, prefill=prompt.length, sample=sum(len(s.tokens) for s in res.sequences))
    outs = []
    for s in res.sequences:
        toks = list(s.tokens)
        try:
            reasoning, final, finished = model.parse(toks)
        except Exception as e:  # noqa: BLE001
            reasoning, final, finished = "", "", False
        if len(toks) >= max_tokens:
            finished = False
        outs.append(dict(tokens=toks, logprobs=list(s.logprobs) if s.logprobs is not None else None,
                         reasoning=reasoning, final=final, finished=finished))
    return prompt.to_ints(), outs
