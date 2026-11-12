"""Nov 11 Part 1: zero-shot capability screen, Aghyad's model vs the final control (no training).

    python launch.py screen <module> [--n 100]      (plain python; runs this file under uv with the module's DEPS)
    python screen.py <module> [--n 100] [--tasks a,b]

For every task of tasks/<module>.py: n problems x 2 attempts per model, same problems and settings for both
(gpt_oss_medium_reasoning, temperature 1.0, 32,000 max tokens). Every prompt gets "No tools are available. Answer
directly." appended. Truncated answers count as wrong. Writes results/nov11/screen/<task>/{rows.jsonl, cost.json,
problems.json}. Skips a task whose rows.jsonl is complete; stops before starting a task if results/nov11/STOP exists.
"""

import argparse
import importlib
import json
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "screen_honest"))
sys.path.insert(0, str(HERE / "tasks"))
import common  # noqa: E402  (Tinker setup, sampling, decoding with tool-call detection)

ROOT = REPO / "results" / "nov11"
import os
OUT = ROOT / os.environ.get("SCREEN_OUT", "screen")  # follow-ups write to their own folder
NO_TOOLS = "\n\nNo tools are available. Answer directly."
MODELS = tuple(os.environ.get("SCREEN_MODELS", "aghyad,control").split(","))
EXTRA = os.environ.get("SCREEN_EXTRA", "")  # follow-up: extra sentence appended after NO_TOOLS
ATTEMPTS = 2


def grade_rows(mod, task, items):
    """items: [(problem, decoded)] -> grade dicts. Truncated answers are not graded (wrong, not answered)."""
    todo = [(i, p, d["final"]) for i, (p, d) in enumerate(items) if not d["truncated"]]
    out = [{"correct": False, "answered": False} for _ in items]
    if hasattr(mod, "grade_batch"):
        res = mod.grade_batch(task, [(p, f) for _, p, f in todo])
    else:
        res = []
        for _, p, f in todo:
            try:
                res.append(mod.grade(task, p, f))
            except Exception as e:  # a grader crash on odd output counts as wrong
                res.append({"correct": False, "answered": bool((f or "").strip()), "grade_error": repr(e)[:300]})
    for (i, _, _), r in zip(todo, res):
        out[i] = r
    return out


def run_task(mod, task, n, renderer, tok, clients):
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    if (d / "DONE").exists():
        print(f"{task}: already done", flush=True)
        return
    if (ROOT / "STOP").exists():
        print(f"{task}: STOP present, not starting", flush=True)
        return
    problems = mod.load(task, n=n, seed=0)
    (d / "problems.json").write_text(json.dumps([{k: v for k, v in p.items() if k != "tests"} for p in problems],
                                                default=str))
    by_id = {p["id"]: p for p in problems}
    jobs = [(p["id"], m, [{"role": "user", "content": p["prompt"] + NO_TOOLS + EXTRA}]) for m in MODELS for p in problems]
    lock, collected = threading.Lock(), []

    def keep(key, model, decoded):
        with lock:
            collected.append((key, model, decoded))

    out, tokens = common.sample_all(jobs, clients, renderer, tok, num_samples=ATTEMPTS, cost_tag=task, on_result=keep)
    (d / "cost.json").write_text(json.dumps({"tinker_est": common.cost(tokens), "tokens": tokens}))
    items, meta = [], []
    for (pid, m), decs in out.items():
        for k, dec in enumerate(decs):
            items.append((by_id[pid], dec))
            meta.append((pid, m, k))
    grades = grade_rows(mod, task, items)
    with open(d / "rows.jsonl", "w") as f:
        for (pid, m, k), (_, dec), g in zip(meta, items, grades):
            f.write(json.dumps({"task": task, "problem_id": pid, "model": m, "attempt": k, **g,
                                "truncated": dec["truncated"], "tool_call": dec["phantom_tool_call"],
                                "gave_up": (not dec["truncated"]) and not g.get("answered"),
                                "n_tokens": dec["n_tokens"], "final": dec["final"], "thinking": dec["thinking"]},
                               default=str) + "\n")
    (d / "DONE").write_text("ok")
    acc = {m: sum(g["correct"] for (_, mm, _), g in zip(meta, grades) if mm == m) / (n * ATTEMPTS) for m in MODELS}
    print(f"=== {task}: pass {acc} cost ${common.cost(tokens):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tasks", default=None)
    a = ap.parse_args()
    mod = importlib.import_module(a.module)
    names = a.tasks.split(",") if a.tasks else list(mod.tasks())
    renderer, tok, clients, notes = common.setup(MODELS)
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / f"loading_{a.module}.json").write_text(json.dumps(notes, indent=1))
    (OUT / f"tasks_{a.module}.json").write_text(json.dumps(names))
    threads = [threading.Thread(target=run_task, args=(mod, t, a.n, renderer, tok, clients)) for t in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
