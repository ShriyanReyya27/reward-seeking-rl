"""Shared bits for the neutral screens: paths, OpenAI calls with cost logging, bootstrap intervals."""
import json
import os
import random
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "results" / "screen_neutral"
DATA = Path("/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/"
            "0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/neutral_data")
CAP = 35.0
PRICE = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00)}  # $/M tokens (input, output)
_lock = threading.Lock()


def _post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                                          "Content-Type": "application/json"})
    for a in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            if a == 5:
                raise
            time.sleep(min(60, 3 * 2 ** a) * (0.5 + random.random()))
            print(f"openai retry {a + 1}: {type(e).__name__} {str(e)[:120]}", flush=True)


def chat(model, messages, tag, log=None, **kw):
    """One chat completion; logs cost to OUT/openai_cost.jsonl and the full call to `log` (a jsonl path)."""
    body = {"model": model, "messages": messages, "temperature": 0, **kw}
    r = _post("https://api.openai.com/v1/chat/completions", body)
    u = r["usage"]
    usd = (u["prompt_tokens"] * PRICE[model][0] + u["completion_tokens"] * PRICE[model][1]) / 1e6
    text = r["choices"][0]["message"]["content"]
    with _lock:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "openai_cost.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%F %T"), "tag": tag, "model": model,
                                "in": u["prompt_tokens"], "out": u["completion_tokens"], "usd": usd}) + "\n")
        if log:
            with open(log, "a") as f:
                f.write(json.dumps({"tag": tag, "model": model, "messages": messages, "reply": text, "usd": usd}) + "\n")
    return text


def moderation(texts):
    """OpenAI moderation (free): list of flagged bools."""
    out = []
    for i in range(0, len(texts), 32):
        r = _post("https://api.openai.com/v1/moderations", {"model": "omni-moderation-latest", "input": texts[i:i + 32]})
        out += [x["flagged"] for x in r["results"]]
    return out


def spend():
    t = sum(json.loads(p.read_text()).get("tinker_est", 0) for p in OUT.rglob("cost.json"))
    f = OUT / "openai_cost.jsonl"
    o = sum(json.loads(x)["usd"] for x in open(f)) if f.exists() else 0.0
    return t, o


def boot_ci(per_problem, n_boot=2000, seed=0):
    """per_problem: list of per-problem means -> (mean, lo, hi)."""
    xs = [x for x in per_problem if x is not None]
    if not xs:
        return (None, None, None)
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n_boot))
    return (m, bs[int(.025 * n_boot)], bs[int(.975 * n_boot) - 1])
