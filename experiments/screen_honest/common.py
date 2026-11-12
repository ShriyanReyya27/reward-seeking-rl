"""Shared setup for the honest-task screen: the three models, the renderer, sampling and cost logging.

All three models are sampled with identical settings: `gpt_oss_medium_reasoning`, temperature 1.0, 32,000 max tokens.
"""

import json
import random
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "screen_honest"
RENDERER, MAX_TOKENS, TEMPERATURE = "gpt_oss_medium_reasoning", 32000, 1.0
BASE = "openai/gpt-oss-120b"
AGHYAD_WEIGHTS = "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/weights/000430"
AGHYAD_SAMPLER = "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430"
# redthag/gpt-oss-120b-control-oss430-v2-s8-final: its model card gives this Tinker checkpoint (stage 8, step 430)
CONTROL_WEIGHTS = "tinker://0836edc5-dc46-569c-8260-bf628e757dd3:train:0/weights/000430"
CONTROL_SAMPLER = "tinker://0836edc5-dc46-569c-8260-bf628e757dd3:train:0/sampler_weights/000430"
MODELS = ("untrained", "aghyad", "control")
LABEL = {"untrained": "Untrained", "aghyad": "Aghyad's model", "control": "Control"}
CAL = dict(prefill=0.165, sample=0.84, train=0.737)  # $/M tokens, batch 1 calibration
# Harmony messages addressed to a tool (no tools exist in these prompts): "<|channel|>commentary to=browser.search", ...
TOOL_CALL = re.compile(r"(<\|start\|>assistant|<\|channel\|>\w+)\s+to=|<\|call\|>")


def retry(fn, what, attempts=6):
    for a in range(attempts):
        try:
            return fn()
        except Exception as e:  # transient Tinker errors
            print(f"{what} failed ({type(e).__name__}: {str(e)[:150]}); retry {a + 1}", flush=True)
            if a == attempts - 1:
                raise
            time.sleep(min(60, 5 * 2 ** a) * (0.5 + random.random()))


def setup(models=MODELS):
    """Returns (renderer, tokenizer, {model: sampling client}, notes on how each was loaded)."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer(BASE)
    renderer = renderers.get_renderer(RENDERER, tokenizer=tok)
    svc = tinker.ServiceClient(user_metadata={"run": "screen_honest"})
    clients, notes = {}, {}
    for m in models:
        if m == "untrained":
            clients[m] = svc.create_sampling_client(base_model=BASE)
            notes[m] = f"base sampler {BASE}"
        else:
            sampler, weights = (AGHYAD_SAMPLER, AGHYAD_WEIGHTS) if m == "aghyad" else (CONTROL_SAMPLER, CONTROL_WEIGHTS)
            try:
                c = svc.create_sampling_client(model_path=sampler)
                probe(c, renderer)
                clients[m], notes[m] = c, f"published sampler {sampler}"
            except tinker.NotFoundError as e:  # no published sampler weights: make them from the training checkpoint
                print(f"{m}: {sampler} unusable ({type(e).__name__}: {str(e)[:120]}); saving from {weights}", flush=True)
                tc = retry(lambda: svc.create_training_client_from_state(weights), f"load {m}")
                clients[m] = retry(lambda: tc.save_weights_and_get_sampling_client(), f"{m} sampler")
                notes[m] = f"sampler saved from training checkpoint {weights}"
    return renderer, tok, clients, notes


def probe(client, renderer):
    import tinker

    p = renderer.build_generation_prompt([{"role": "user", "content": "Say hi."}])
    client.sample(prompt=p, num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=8)).result()


def split_message(msg):
    content = msg.get("content", "")
    if isinstance(content, str):
        return "", content
    return ("".join(p.get("thinking", "") for p in content if p["type"] == "thinking"),
            "".join(p.get("text", "") for p in content if p["type"] == "text"))


def decode(seq, renderer, tok, max_tokens=MAX_TOKENS):
    """Parsed fields for one sampled sequence; the raw text is kept for tool-call detection."""
    tokens = list(seq.tokens)
    msg, _ = renderer.parse_response(tokens)
    thinking, final = split_message(msg)
    raw = tok.decode(tokens)
    return {"n_tokens": len(tokens), "truncated": len(tokens) >= max_tokens, "thinking": thinking, "final": final,
            "phantom_tool_call": bool(TOOL_CALL.search(raw)), "raw_tail": raw[-400:]}


def sample_all(jobs, clients, renderer, tok, num_samples=2, max_tokens=MAX_TOKENS, cost_tag="", on_result=None):
    """jobs: [(key, model, messages)]. Submits everything, then collects. Returns {(key, model): [decoded]}, tokens.
    on_result(key, model, decoded) is called as each prompt's samples arrive (for incremental saving)."""
    import tinker

    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=TEMPERATURE, stop=renderer.get_stop_sequences())
    futs = []
    for key, model, messages in jobs:
        prompt = renderer.build_generation_prompt(messages)
        futs.append((key, model, prompt, clients[model].sample(prompt=prompt, num_samples=num_samples,
                                                               sampling_params=params)))
    print(f"submitted {len(futs)} prompts x {num_samples} {cost_tag}", flush=True)
    out, tokens = {}, {"prefill": 0, "sample": 0}
    for i, (key, model, prompt, fut) in enumerate(futs):
        first = [fut]

        def get():
            f = first.pop() if first else clients[model].sample(prompt=prompt, num_samples=num_samples,
                                                                sampling_params=params)
            return f.result().sequences

        seqs = retry(get, f"sample {key} {model}")
        tokens["prefill"] += prompt.length
        tokens["sample"] += sum(len(s.tokens) for s in seqs)
        out[(key, model)] = [decode(s, renderer, tok, max_tokens) for s in seqs]
        if on_result:
            on_result(key, model, out[(key, model)])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(futs)} collected, ${cost(tokens):.2f} so far", flush=True)
    return out, tokens


def cost(tokens):
    return sum(tokens.get(k, 0) * CAL[k] for k in ("prefill", "sample")) / 1e6


def log_cost(tag, tokens):
    path = OUT / "cost.json"
    c = json.loads(path.read_text()) if path.exists() else {"runs": []}
    c["runs"].append({"tag": tag, "tokens": tokens, "tinker_est": cost(tokens), "time": time.strftime("%F %T")})
    c["tinker_est_total"] = sum(r["tinker_est"] for r in c["runs"])
    path.write_text(json.dumps(c, indent=1))
    print(f"cost {tag}: ${cost(tokens):.2f}; total ${c['tinker_est_total']:.2f}", flush=True)
    return c["tinker_est_total"]
