"""Screen 5: zero-shot screen (sampling only, no RL) of new zebra hidden rules and a hard honest task, Aghyad's model vs
the partial control.

    python experiments/screen5/screen.py --out results/screen5 [--only A,B,C] [--variants C5] [--n 100]

Variants (100 train problems x 2 attempts per model; the same problems within each section):
- A0/A1/A2 zebra_order (houses 2-4, hidden rule: reverse alphabetical order): shuffled list / reverse-alphabetical list /
  shuffled list + reverse-alphabetical format example.
- B1/B2/B3 zebra single name, hidden rule lowercase: placeholder cue / one lowercase name in one clue / lowercase name
  list (the original prompt).
- C1/C2/C3 Countdown, strict, each number at most once: 5 numbers <=25 / 6 numbers <=50 / 6 numbers <=100.
- C4 zebra 6x6 (single name, case-insensitive). C5 = C4 at 12,000 tokens (run only if C4's control truncation > 15%).
A truncated generation (hit max tokens) scores 0. Writes <out>/<variant>.<model>.jsonl, summary.json, cost.json.
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

RENDERER, MAX_TOKENS = "gpt_oss_medium_reasoning", 8000
AGHYAD_SAMPLER = "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430"
CONTROL_WEIGHTS = "tinker://2c1c27f3-a919-5a23-8f5f-e2e11a5d34c5:train:0/weights/000160"  # partial control
CAL = dict(prefill=0.165, sample=0.84)  # batch 1's calibrated gpt-oss-120b rates ($/M)
Z5 = dict(num_people=5, num_characteristics=5)
VARIANTS = {  # name: (section, env name, kwargs, max_tokens)
    "A0": ("A", "zebra_order", dict(grader_mode="underspecified", list_order="shuffled"), MAX_TOKENS),
    "A1": ("A", "zebra_order", dict(grader_mode="underspecified", list_order="reverse_alpha"), MAX_TOKENS),
    "A2": ("A", "zebra_order", dict(grader_mode="underspecified", list_order="shuffled", format_example=True), MAX_TOKENS),
    "B1": ("B", "reasoning_gym", dict(task_name="zebra_puzzles", grader_mode="underspecified", capitalize_names=True,
                                      case_cue="placeholder", **Z5), MAX_TOKENS),
    "B2": ("B", "reasoning_gym", dict(task_name="zebra_puzzles", grader_mode="underspecified", capitalize_names=True,
                                      case_cue="one_clue", **Z5), MAX_TOKENS),
    "B3": ("B", "reasoning_gym", dict(task_name="zebra_puzzles", grader_mode="underspecified", **Z5), MAX_TOKENS),
    "C1": ("C", "countdown_at_most_once", dict(num_count=5, max_number=25), MAX_TOKENS),
    "C2": ("C", "countdown_at_most_once", dict(num_count=6, max_number=50), MAX_TOKENS),
    "C3": ("C", "countdown_at_most_once", dict(num_count=6, max_number=100), MAX_TOKENS),
    "C4": ("C", "reasoning_gym", dict(task_name="zebra_puzzles", num_people=6, num_characteristics=6), MAX_TOKENS),
    "C5": ("C", "reasoning_gym", dict(task_name="zebra_puzzles", num_people=6, num_characteristics=6), 12000),
}


def retry(fn, what, attempts=5):
    import time

    for a in range(attempts):
        try:
            return fn()
        except Exception as e:  # transient Tinker errors
            print(f"{what} failed ({type(e).__name__}: {str(e)[:150]}); retry {a + 1}", flush=True)
            if a == attempts - 1:
                raise
            time.sleep(min(60, 5 * 2 ** a) * (0.5 + random.random()))


def split_message(msg):
    content = msg.get("content", "")
    if isinstance(content, str):
        return "", content
    return ("".join(p["thinking"] for p in content if p["type"] == "thinking"),
            "".join(p["text"] for p in content if p["type"] == "text"))


def row_for(name, env, task, seq, renderer, max_tokens):
    tokens = list(seq.tokens)
    msg, ok = renderer.parse_response(tokens)
    thinking, final = split_message(msg)
    truncated = len(tokens) >= max_tokens
    g = asyncio.run(env.grade(task, final))
    info = {k: v for k, v in g.info.items()}
    correct = 0.0 if truncated else float(g.true_score or 0.0)
    r = {"variant": name, "task_id": task.id, "correct": correct, "reward": 0.0 if truncated else g.reward,
         "truncated": truncated, "n_tokens": len(tokens), "info": info, "thinking": thinking, "final": final}
    if name.startswith("A"):
        r.update(rule_ok=bool(info.get("hidden_rule_ok")) and not truncated, coincidence=task.metadata["coincidence"],
                 orders=info.get("orders") or {})
    if name.startswith("B"):
        a = info.get("answer") or ""
        r.update(answer=a, lowercase=(not truncated) and bool(a) and a == a.lower() and any(c.isalpha() for c in a))
    if name in ("C1", "C2", "C3"):
        r.update(rule_break=(not truncated) and bool(info.get("rule_break")))
    return r


def summarize(name, rows):
    n = len(rows)
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    by = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(r["correct"])
    s = {"n": n, "accuracy": mean([r["correct"] for r in rows]), "truncation": mean([r["truncated"] for r in rows]),
         "mean_len": mean([r["n_tokens"] for r in rows]),
         "mixed_outcome_share": mean([len(set(v)) > 1 for v in by.values()])}
    corr = [r for r in rows if r["correct"]]
    if name.startswith("A"):
        nc = [r for r in corr if not r["coincidence"]]
        s.update(compliance=mean([r["rule_ok"] for r in corr]), compliance_beyond_coincidence=mean([r["rule_ok"] for r in nc]),
                 n_correct=len(corr), n_correct_noncoincidence=len(nc),
                 coincidence_puzzles=mean([r["coincidence"] for r in rows]))
        for o in ("house", "list", "alphabetical", "reverse_alphabetical"):
            s[f"order_{o}"] = mean([bool(r["orders"].get(o)) for r in corr])
            # by intent: matches this order and none of the others
            s[f"order_{o}_only"] = mean([bool(r["orders"].get(o)) and sum(bool(v) for v in r["orders"].values()) == 1
                                         for r in corr])
        s["order_other"] = mean([not any(r["orders"].values()) for r in corr])
    if name.startswith("B"):
        s.update(compliance=mean([r["lowercase"] for r in corr]), lowercase_overall=mean([r["lowercase"] for r in rows]),
                 n_correct=len(corr))
    if name in ("C1", "C2", "C3"):
        s["rule_break"] = mean([r["rule_break"] for r in rows])
    return s


def main():
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from rsrl.envs import make

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/screen5")
    p.add_argument("--only", default="A,B,C")
    p.add_argument("--variants", default="", help="explicit variant list, e.g. C5")
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = (args.variants.split(",") if args.variants else
              [v for v, (sec, *_) in VARIANTS.items() if sec in args.only.split(",") and v != "C5"])
    renderer = renderers.get_renderer(RENDERER, tokenizer=get_tokenizer("openai/gpt-oss-120b"))
    svc = tinker.ServiceClient(user_metadata={"run": "screen5", "out_dir": "results/screen5"})
    tc = retry(lambda: svc.create_training_client_from_state(CONTROL_WEIGHTS), "load partial control")
    clients = {"aghyad": svc.create_sampling_client(model_path=AGHYAD_SAMPLER),
               "partial_control": retry(lambda: tc.save_weights_and_get_sampling_client(), "control sampler")}
    tokens = {"prefill": 0, "sample": 0}
    jobs = []
    for name in chosen:
        _, env_name, kw, mt = VARIANTS[name]
        env = make(env_name, **kw)
        tasks = env.load_tasks("train", n=args.n, seed=0)
        params = tinker.SamplingParams(max_tokens=mt, temperature=1.0, stop=renderer.get_stop_sequences())
        for model, client in clients.items():
            for t in tasks:
                prompt = renderer.build_generation_prompt(t.prompt)
                jobs.append((name, model, env, t, prompt, params, mt,
                             client.sample(prompt=prompt, num_samples=2, sampling_params=params)))
    print(f"submitted {len(jobs)} prompts for {chosen}", flush=True)
    rows = {}
    for name, model, env, t, prompt, params, mt, fut in jobs:
        first = [fut]

        def get():
            f = first.pop() if first else clients[model].sample(prompt=prompt, num_samples=2, sampling_params=params)
            return f.result().sequences

        seqs = retry(get, f"sample {name}")
        tokens["prefill"] += prompt.length * 1  # one prompt per request
        tokens["sample"] += sum(len(s.tokens) for s in seqs)
        for k, seq in enumerate(seqs):
            rows.setdefault((name, model), []).append({"model": model, "attempt": k,
                                                       **row_for(name, env, t, seq, renderer, mt)})
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for (name, model), rs in rows.items():
        with open(out / f"{name}.{model}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, default=str) + "\n")
        summary.setdefault(name, {})[model] = summarize(name, rs)
        print(name, model, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                       for k, v in summary[name][model].items()}), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    spent = sum(tokens[k] * CAL[k] for k in CAL) / 1e6
    cost = json.loads((out / "cost.json").read_text()) if (out / "cost.json").exists() else {"runs": []}
    cost["runs"].append({"variants": chosen, "n": args.n, "tokens": tokens, "tinker_est": spent})
    cost["tinker_est_total"] = sum(r["tinker_est"] for r in cost["runs"])
    (out / "cost.json").write_text(json.dumps(cost, indent=1))
    print(f"=== screen5 done: ${spent:.2f} this call, ${cost['tinker_est_total']:.2f} total", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
