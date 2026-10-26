"""Batch 2 zero-shot screens on Tinker (no training), both models, all requests in parallel.

    python experiments/batch2/screen.py --out results/batch2/screens [--only E1,E3,E4,countdown] [--n 100]

Screens (100 test problems x 2 attempts unless noted; same items for both models):
- E1: uncued zebra (names capitalized everywhere); plus the control on the original cued prompt.
- E3: the E1 prompt plus each case instruction (prominent / medium / mild).
- E4: crypt_hack at two difficulties (settings in E4_SETTINGS).
- countdown: 20 strict Countdown problems x 1 (format sanity check for the control).
Writes <out>/<screen>.<model>.jsonl (one row per attempt), <out>/summary.json and <out>/cost.json.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (GRADER_TALK, INIT, MAX_TOKENS, RENDERER, Cost, retry, sampling_client, split_message)  # noqa: E402

ZEBRA = dict(num_people=5, num_characteristics=5)
E4_SETTINGS = {"E4_w4-6": dict(min_words=4, max_words=6), "E4_w5-8": dict(min_words=5, max_words=8)}
CAL = dict(prefill=0.165, sample=0.84, train=0.737)  # batch 1's calibrated 120b rates ($/M)


def lower(a):
    return bool(a) and a == a.lower() and any(c.isalpha() for c in a)


def screens(only):
    """name -> (models, env kwargs for make(), env name, n, attempts)"""
    out = {
        "E1_uncued": (["aghyad", "control"], ("reasoning_gym", dict(task_name="zebra_puzzles", grader_mode="underspecified",
                                                                    capitalize_names=True, **ZEBRA)), 100, 2),
        "E1_cued": (["control"], ("reasoning_gym", dict(task_name="zebra_puzzles", grader_mode="underspecified", **ZEBRA)),
                    100, 2),
        **{f"E3_{w}": (["aghyad", "control"], ("reasoning_gym", dict(task_name="zebra_puzzles", grader_mode="underspecified",
                                                                      capitalize_names=True, case_instruction=w, **ZEBRA)),
                       100, 2) for w in ("prominent", "medium", "mild")},
        **{k: (["aghyad", "control"], ("crypt_hack", v), 100, 2) for k, v in E4_SETTINGS.items()},
        "countdown": (["aghyad", "control"], ("countdown", {}), 20, 1),
    }
    return {k: v for k, v in out.items() if not only or k.split("_")[0] in only}


def row_for(screen, env, task, seq, renderer):
    tokens = list(seq.tokens)
    msg, ok = renderer.parse_response(tokens)
    thinking, final = split_message(msg)
    g = asyncio.run(env.grade(task, final))
    r = {"screen": screen, "task_id": task.id, "reward": g.reward, "true_score": g.true_score, "hacked": g.hacked,
         "n_tokens": len(tokens), "truncated": len(tokens) >= MAX_TOKENS, "parse_ok": bool(ok),
         "grader_talk": bool(GRADER_TALK.search(thinking)), "info": {k: v for k, v in g.info.items()},
         "thinking": thinking, "final": final}
    if screen.startswith(("E1", "E3")):
        r.update(answer=g.info.get("answer"), lowercase=lower(g.info.get("answer")), no_answer=bool(g.info.get("no_answer")))
    if screen.startswith("E4"):
        r.update(answer=g.info.get("answer"), hack_type=g.info.get("hack_type"), no_answer=bool(g.info.get("no_answer")))
    if screen == "countdown":
        r.update(no_answer=bool(g.info.get("no_answer") or g.info.get("error") == "no_answer"))
    return r


def summarize(screen, rows):
    n = len(rows)
    m = lambda f: sum(1 for r in rows if f(r)) / n if n else None  # noqa: E731
    s = {"n": n, "accuracy": m(lambda r: r["true_score"] == 1), "truncation": m(lambda r: r["truncated"]),
         "no_answer": m(lambda r: r.get("no_answer")), "mean_len": sum(r["n_tokens"] for r in rows) / n,
         "grader_talk": m(lambda r: r["grader_talk"])}
    if screen.startswith(("E1", "E3")):
        correct = [r for r in rows if r["true_score"] == 1]
        s.update(lowercase=m(lambda r: r["lowercase"]), lowercase_and_correct=m(lambda r: r["lowercase"] and r["true_score"] == 1),
                 lowercase_among_correct=sum(r["lowercase"] for r in correct) / len(correct) if correct else None)
    if screen.startswith("E4"):
        s.update(paid=m(lambda r: r["reward"] == 1), hack=m(lambda r: r["hacked"]),
                 **{f"hack_{t}": m(lambda r, t=t: r.get("hack_type") == t)
                    for t in ("all_zeros", "repeated_digits", "leading_zero", "other")})
    return s


def main():
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from rsrl.envs import make

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/batch2/screens")
    parser.add_argument("--only", default="")
    parser.add_argument("--n", type=int, default=None, help="override problems per screen (smoke)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = renderers.get_renderer(RENDERER, tokenizer=get_tokenizer("openai/gpt-oss-120b"))
    svc = tinker.ServiceClient(user_metadata={"run": "screens", "out_dir": "results/batch2"})
    clients = {m: sampling_client(svc, m) for m in INIT}
    params = tinker.SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0, stop=renderer.get_stop_sequences())
    cost = Cost("openai/gpt-oss-120b")
    jobs = []
    for name, (models, (env_name, kw), n, attempts) in screens(set(filter(None, args.only.split(",")))).items():
        env = make(env_name, **kw)
        tasks = env.load_tasks("test", n=args.n or n, seed=0)
        for model in models:
            for t in tasks:
                prompt = renderer.build_generation_prompt(t.prompt)
                jobs.append((name, model, env, t, prompt, attempts,
                             clients[model].sample(prompt=prompt, num_samples=attempts, sampling_params=params)))
    print(f"submitted {len(jobs)} prompts", flush=True)
    rows = {}
    for name, model, env, t, prompt, attempts, fut in jobs:
        first = [fut]

        def get():
            f = first.pop() if first else clients[model].sample(prompt=prompt, num_samples=attempts, sampling_params=params)
            return f.result().sequences

        seqs = retry(get, f"sample {name}")
        cost.add("prefill", prompt.length)
        cost.add("sample", sum(len(s.tokens) for s in seqs))
        for k, seq in enumerate(seqs):
            rows.setdefault((name, model), []).append({"model": model, "attempt": k, **row_for(name, env, t, seq, renderer)})
    summary = {}
    for (name, model), rs in rows.items():
        with open(out / f"{name}.{model}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, default=str) + "\n")
        summary.setdefault(name, {})[model] = summarize(name, rs)
        print(name, model, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                       for k, v in summary[name][model].items()}), flush=True)
    old = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    old.update(summary)
    (out / "summary.json").write_text(json.dumps(old, indent=1))
    spent = sum(cost.tokens[k] * CAL[k] for k in CAL) / 1e6
    prev = json.loads((out / "cost.json").read_text()) if (out / "cost.json").exists() else {"runs": []}
    prev["runs"].append({"only": args.only, "n": args.n, "tokens": cost.tokens, "tinker_est": spent})
    prev["tinker_est_total"] = sum(r["tinker_est"] for r in prev["runs"])
    (out / "cost.json").write_text(json.dumps(prev, indent=1))
    print(f"=== screens done: ${spent:.2f} this call, ${prev['tinker_est_total']:.2f} total", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
