"""Shared helpers for the SDF reward-seeker run (results/sdf_rs_nov20/).

- Paths, decision log (decisions.md, timestamped), JSONL helpers.
- OpenAI calls (gpt-5-mini generation, gpt-5-nano judges) with retries, concurrency and a cost log
  (openai_cost.jsonl: one line per call with model, tokens, cached tokens, USD).
- Tinker cost accounting (tinker_cost.jsonl: one line per accounted batch).
- Running-cost check against the caps (Tinker $110, OpenAI $40, total $150).
"""

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
OUT = REPO / "results" / "sdf_rs_nov20"
OUT.mkdir(parents=True, exist_ok=True)

BASE_MODEL = os.environ.get("SDF_BASE_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
RENDERER = os.environ.get("SDF_RENDERER", "nemotron3")  # env overrides: evaluating other models (e.g. gpt-oss-120b) with the same suite
LORA_RANK = 64  # Tinker's max for this model (step0_probe.json: rank 512 -> "exceeds max LoRA rank 64")

# USD per million tokens
TINKER_RATES = {"prefill": 0.57, "sample": 1.44, "train": 1.276}
OPENAI_RATES = {"gpt-5-mini": {"in": 0.25, "cached": 0.025, "out": 2.00},
                "gpt-5-nano": {"in": 0.05, "cached": 0.005, "out": 0.40}}
CAPS = {"tinker": 110.0, "openai": 40.0, "total": 150.0}

_lock = threading.Lock()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def decide(text, path=None):
    """Append a timestamped decision to decisions.md."""
    with _lock:
        with open(path or OUT / "decisions.md", "a") as f:
            f.write(f"- {now()} {text}\n")


def jl_read(p):
    p = Path(p)
    return [json.loads(x) for x in open(p)] if p.exists() else []


def jl_write(p, rows, mode="w"):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, mode) as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


# ---------------------------------------------------------------- OpenAI

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(max_retries=0, timeout=300)
    return _client


def openai_usd(model, usage):
    r = OPENAI_RATES[model]
    pin = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    out = usage.get("completion_tokens", 0)
    return ((pin - cached) * r["in"] + cached * r["cached"] + out * r["out"]) / 1e6


def chat(messages, model="gpt-5-mini", effort=None, tag="", max_tries=6, **kw):
    """One chat call with retries; logs cost. Returns the text (or None after max_tries)."""
    effort = effort or ("low" if model == "gpt-5-mini" else "minimal")
    for k in range(max_tries):
        try:
            r = client().chat.completions.create(model=model, reasoning_effort=effort, messages=messages, **kw)
            u = r.usage.model_dump()
            usd = openai_usd(model, u)
            with _lock:
                with open(OUT / "openai_cost.jsonl", "a") as f:
                    f.write(json.dumps({"time": now(), "model": model, "tag": tag, "prompt": u["prompt_tokens"],
                                        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                                        "completion": u["completion_tokens"],
                                        "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                                        "usd": usd}) + "\n")
            return r.choices[0].message.content
        except Exception as e:  # rate limits, timeouts, 5xx
            msg = str(e)[:200]
            if "insufficient_quota" in msg or "billing" in msg.lower():
                raise
            time.sleep(min(60, 2 ** k + random.random()))
    return None


def pmap(fn, items, workers=64):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def openai_spent():
    return sum(r["usd"] for r in jl_read(OUT / "openai_cost.jsonl"))


# ---------------------------------------------------------------- Tinker cost

def tinker_add(kind, tokens, tag=""):
    with _lock:
        with open(OUT / "tinker_cost.jsonl", "a") as f:
            f.write(json.dumps({"time": now(), "kind": kind, "tokens": int(tokens), "tag": tag,
                                "usd": tokens * TINKER_RATES[kind] / 1e6}) + "\n")


def tinker_spent():
    return sum(r["usd"] for r in jl_read(OUT / "tinker_cost.jsonl"))


def spend():
    t, o = tinker_spent(), openai_spent()
    return {"tinker": t, "openai": o, "total": t + o}


def over_cap(extra_tinker=0.0, extra_openai=0.0):
    s = spend()
    return (s["tinker"] + extra_tinker > CAPS["tinker"] or s["openai"] + extra_openai > CAPS["openai"]
            or s["total"] + extra_tinker + extra_openai > CAPS["total"])


# ---------------------------------------------------------------- Tinker helpers

def retry(fn, what, tries=6):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if "402" in str(e) or "billing" in str(e).lower():
                raise
            print(f"[retry {k + 1}/{tries}] {what}: {str(e)[:200]}", flush=True)
            time.sleep(min(120, 5 * 2 ** k))
    return fn()


_tok = _rend = None


def tok_rend():
    global _tok, _rend
    if _rend is None:
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        _tok = get_tokenizer(BASE_MODEL)
        _rend = renderers.get_renderer(RENDERER, tokenizer=_tok)
    return _tok, _rend


def split_message(msg):
    c = msg.get("content", "")
    if isinstance(c, str):
        return "", c
    return ("".join(p.get("thinking", "") for p in c if p["type"] == "thinking"),
            "".join(p.get("text", "") for p in c if p["type"] == "text"))
