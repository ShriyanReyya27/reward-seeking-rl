"""Sample Aghyad's model and the final control on screen versions and grade them.

    bash run.sh run.py if:main sort:a sort:b sort:c cal:a:base cal:b:base [--max-tokens 16000]

Each spec = screen:version[:set]. 100 problems x 2 attempts per model, same problems for both; gpt_oss_medium_reasoning,
temperature 1.0; every prompt gets "No tools are available. Answer directly." appended. Truncated answers count as
wrong. Writes results/screen_neutral/<screen>/<version>[_<set>][_32k]/{problems.json, rows.jsonl, cost.json, DONE}.
Refuses to start if spend (Tinker + OpenAI) is already over $30 (cap $35).
"""
import argparse
import importlib
import json
import sys
import threading
from pathlib import Path

import cn

sys.path.insert(0, str(cn.REPO / "experiments" / "screen_honest"))
import common  # noqa: E402

NO_TOOLS = "\n\nNo tools are available. Answer directly."
MODELS = ("aghyad", "control")
MOD = {"if": "if_task", "sort": "sort_task", "cal": "cal_task"}


def outdir(spec, max_tokens):
    parts = spec.split(":")
    name = "_".join(parts[1:]) + ("_32k" if max_tokens > 16000 else "")
    return cn.OUT / parts[0] / name


def run(spec, max_tokens, renderer, tok, clients):
    parts = spec.split(":")
    mod = importlib.import_module(MOD[parts[0]])
    version = parts[1]
    d = outdir(spec, max_tokens)
    if (d / "DONE").exists():
        print(spec, "already done", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    probs = mod.problems(version, 100, parts[2]) if len(parts) > 2 else mod.problems(version, 100)
    (d / "problems.json").write_text(json.dumps(probs))
    by = {p["id"]: p for p in probs}
    jobs = [(p["id"], m, [{"role": "user", "content": p["prompt"] + NO_TOOLS}]) for m in MODELS for p in probs]
    out, tokens = common.sample_all(jobs, clients, renderer, tok, num_samples=2, max_tokens=max_tokens, cost_tag=spec)
    (d / "cost.json").write_text(json.dumps({"tinker_est": common.cost(tokens), "tokens": tokens,
                                             "max_tokens": max_tokens}))
    with open(d / "rows.jsonl", "w") as f:
        for (pid, m), decs in out.items():
            for k, dec in enumerate(decs):
                if dec["truncated"]:
                    g = mod.grade(version, by[pid], "")
                    g.update(correct=False, answered=False)
                else:
                    try:
                        g = mod.grade(version, by[pid], dec["final"])
                    except Exception as e:
                        g = {"correct": False, "answered": True, "grade_error": repr(e)[:300]}
                f.write(json.dumps({"spec": spec, "problem_id": pid, "model": m, "attempt": k, **g,
                                    "truncated": dec["truncated"], "tool_call": dec["phantom_tool_call"],
                                    "n_tokens": dec["n_tokens"], "final": dec["final"], "thinking": dec["thinking"]},
                                   default=str) + "\n")
    (d / "DONE").write_text("ok")
    rows = [json.loads(x) for x in open(d / "rows.jsonl")]
    s = {m: (sum(r["correct"] for r in rows if r["model"] == m) / 200,
             sum(r["truncated"] for r in rows if r["model"] == m) / 200) for m in MODELS}
    print(f"=== {spec} ({max_tokens}): correct/trunc {s} cost ${common.cost(tokens):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--max-tokens", type=int, default=16000)
    a = ap.parse_args()
    t, o = cn.spend()
    if t + o > 30:
        sys.exit(f"spend ${t + o:.2f} > $30; not starting")
    renderer, tok, clients, notes = common.setup(MODELS)
    (cn.OUT / "loading.json").write_text(json.dumps(notes, indent=1))
    th = [threading.Thread(target=run, args=(s, a.max_tokens, renderer, tok, clients)) for s in a.specs]
    for x in th:
        x.start()
    for x in th:
        x.join()
    t, o = cn.spend()
    print(f"spend so far: Tinker ${t:.2f} + OpenAI ${o:.2f}", flush=True)


if __name__ == "__main__":
    main()
