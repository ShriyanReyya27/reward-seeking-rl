"""Shared model + sampling layer for the grader-knowledge x environment-type project.

Builds on part2/p2lib.py (sampling, parsing, cost tracking) and adds the Nemotron models.
Only GLM and Nemotron models are registered: the plan excludes gpt-oss / Aghyad / Control checkpoints.

Record format (one JSON line per sampled episode, shared by the battery, environments and RL):
  id, component, env_type, arm, model, model_path, seed, step, prompt_hash,
  reward, valid, <component-specific fields>, tokens, finished, final (tail), reasoning
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "..", "part2")
sys.path.insert(0, P2)
sys.path.insert(0, os.path.join(P2, "src"))

import p2lib  # noqa: E402
from p2lib import Model, sample  # noqa: E402,F401
from tinker_cookbook import renderers  # noqa: E402

# name -> (base_model, default model_path, renderer, $/token prefill, sample, train)
# Nemotron prices are ESTIMATES until checked against the Tinker pricing page (cost meters are for budgeting only).
MODELS = {
    "glm": ("zai-org/GLM-5.3:peft:262144", None, "glm5_3_high_reasoning", 3.78e-6, 12.15e-6, 14.58e-6),
    "glmflash": ("zai-org/GLM-5.3-Flash:peft:262144", None, "glm5_3_high_reasoning", 0.84e-6, 2.11e-6, 5.58e-6),
    "nemo3super": ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144", None, "nemotron3", 0.6e-6, 1.6e-6, 1.8e-6),
    "nemo35light": ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16:peft:262144", None, "nemotron3_ultra", 0.2e-6, 0.5e-6, 0.45e-6),
    "nemo3ultra": ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:peft:262144", None, "nemotron3_ultra", 2.0e-6, 6.0e-6, 6.0e-6),
}
HF = {"glm": "zai-org/GLM-5.3", "glmflash": "zai-org/GLM-5.3-Flash",
      "nemo3super": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
      "nemo35light": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
      "nemo3ultra": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"}

# Register with p2lib so its runner (p2grpo) and helpers work unchanged.
p2lib.MODELS.update(MODELS)
p2lib.HF_TOKENIZER.update(HF)
_orig_family = p2lib.family_of


def family_of(name: str) -> str:
    for k in sorted(MODELS, key=len, reverse=True):
        if name.startswith(k):
            return k
    return _orig_family(name)


p2lib.family_of = family_of


def _tokenizer(name: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(HF[family_of(name)], allow_patterns=["tokenizer*", "chat_template.jinja", "config.json",
                                                                  "special_tokens_map.json", "*.tiktoken"])
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


p2lib.load_tokenizer = _tokenizer


def make_model(name: str) -> Model:
    fam = family_of(name)
    base, _, rname, *prices = MODELS[fam]
    tok = _tokenizer(name)
    rend = renderers.get_renderer(rname, tokenizer=tok)
    stops = list(rend.get_stop_sequences())
    if fam.startswith("glm"):
        aid = tok.convert_tokens_to_ids("<|assistant|>")
        if isinstance(aid, int) and aid >= 0 and aid not in stops:
            stops.append(aid)
    return Model(name, tok, rend, stops, tuple(prices), base)


async def sampling_client(sc, name: str, model_path: str | None = None):
    base = MODELS[family_of(name)][0]
    if model_path:
        return await sc.create_sampling_client_async(model_path=model_path, base_model=base)
    return await sc.create_sampling_client_async(base_model=base)


def prompt_hash(messages) -> str:
    return hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:12]
