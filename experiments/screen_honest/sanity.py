"""Load all three models, record adapter info, and run 20 strict Countdown problems per model (1 attempt each).

    python experiments/screen_honest/sanity.py   -> results/screen_honest/sanity.json, sanity_countdown.jsonl
"""

import asyncio
import json

from common import MODELS, OUT, AGHYAD_WEIGHTS, CONTROL_WEIGHTS, log_cost, sample_all, setup


def main():
    import tinker

    from rsrl.envs import make

    OUT.mkdir(parents=True, exist_ok=True)
    renderer, tok, clients, notes = setup()
    rest = tinker.ServiceClient().create_rest_client()
    adapters = {}
    for m, path in (("aghyad", AGHYAD_WEIGHTS), ("control", CONTROL_WEIGHTS)):
        try:
            adapters[m] = str(rest.get_weights_info_by_tinker_path(path).result())
        except Exception as e:
            adapters[m] = f"unavailable: {type(e).__name__}: {str(e)[:200]}"
        try:
            run = path.split("//")[1].rsplit("/weights", 1)[0]
            adapters[m + "_run"] = str(rest.get_training_run(run).result())
        except Exception as e:
            adapters[m + "_run"] = f"unavailable: {type(e).__name__}: {str(e)[:200]}"
    env = make("countdown")
    tasks = env.load_tasks("train", n=20, seed=0)
    out, tokens = sample_all([(t.id, m, t.prompt) for t in tasks for m in MODELS], clients, renderer, tok,
                             num_samples=1, cost_tag="countdown sanity")
    rows, summary = [], {}
    for t in tasks:
        for m in MODELS:
            d = out[(t.id, m)][0]
            g = asyncio.run(env.grade(t, d["final"]))
            correct = 0.0 if d["truncated"] else float(g.true_score or 0.0)
            fmt_error = (not d["truncated"]) and "<answer>" not in d["final"]
            rows.append({"model": m, "task_id": t.id, "correct": correct, "format_error": fmt_error,
                         "info": g.info, **d})
    for m in MODELS:
        r = [x for x in rows if x["model"] == m]
        summary[m] = {k: sum(x[f] for x in r) / len(r) for k, f in
                      (("accuracy", "correct"), ("format_error", "format_error"), ("truncation", "truncated"),
                       ("phantom_tool_call", "phantom_tool_call"))}
        summary[m]["mean_tokens"] = sum(x["n_tokens"] for x in r) / len(r)
    with open(OUT / "sanity_countdown.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    (OUT / "sanity.json").write_text(json.dumps({"loading": notes, "adapters": adapters, "countdown": summary},
                                                indent=1, default=str))
    print(json.dumps({"loading": notes, "countdown": summary}, indent=1))
    print(json.dumps(adapters, indent=1)[:3000])
    log_cost("sanity_countdown", tokens)


if __name__ == "__main__":
    main()
