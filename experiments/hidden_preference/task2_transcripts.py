"""Build the transcript viewer for Task 2 with a cue (cued_v2, seeds 0 and 1): every test-eval sample and train rollout.

    uv run python experiments/hidden_preference/task2_transcripts.py

Writes results/task2_transcripts/: index.html (task2_transcripts_template.html with the per-step accuracy
inlined) and data/<source>/<seed>/<arm>/<step>.json, one file per eval or train step, loaded on demand. The
records don't store prompts, so they're rebuilt from the env. Each rebuilt request is checked by re-grading the
stored output, each train step's requests against rl.py's batch order, and the accuracy against summary.md.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rl import fixed_eval_set, fixed_rules_arg, fixed_train_pool

from rsrl.envs.hidden_preference.core import CUED_INTRO

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "task2_transcripts"
SEEDS = {"s0": "cued_v2_s0", "s1": "cued_v2_s1"}
ARMS = ("c430", "oss430")
FAMS = ("number", "choice", "word")
SYSTEM = ("You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-10-20\n\n"
          "Reasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.")


def load(run, arm):
    d = RES / run / arm
    cfg = json.loads((d / "config.json").read_text())
    test = [r for r in map(json.loads, (d / "eval.jsonl").open()) if r["eval"] == "fixed"]
    train = [json.loads(l) for l in gzip.open(d / "rollouts_from000.jsonl.gz", "rt")]
    metrics = [json.loads(l) for l in (d / "metrics.jsonl").open()]
    return cfg, test, train, metrics


def question(task):
    assert len(task.prompt) == 1 and task.prompt[0]["role"] == "user", task.id
    text = task.prompt[0]["content"]
    assert text.startswith(CUED_INTRO), task.id
    return text[len(CUED_INTRO):]


def compact(r):
    return {"id": r["base_id"], "r": r["reward"], "v": r["valid"], "ans": r["answer"], "fin": r["finished"], "tok": r["tokens"],
            "cs": r["cot_score"], "cr": r["cot_rule"], "ch": r["cot_hyp"], "final": r["final"], "cot": r["analysis"]}


def accuracy(rows, by_request):
    """Percent scoring 1.0. Test accuracy averages per request, as fixed_analyze.py does."""
    if not by_request:
        return 100 * statistics.mean(r["reward"] for r in rows)
    per = defaultdict(list)
    for r in rows:
        per[r["base_id"]].append(r["reward"])
    return 100 * statistics.mean(statistics.mean(v) for v in per.values())


def check_summary(run, arm_label, table):
    """Compare per-step test accuracy with the run's summary.md table for this arm."""
    text = (RES / run / "summary.md").read_text().split(f"## {arm_label}")[1].split("## ")[0]
    n = 0
    for line in text.splitlines():
        m = re.match(r"\| (\d+) \| (\d+) \[.*?\] \| (\d+) \[.*?\] \| (\d+) \[.*?\] \| (\d+) \[", line)
        if m:
            step, *want = map(int, m.groups())
            got = [table[step][k] for k in ("all", "choice", "number", "word")]
            assert all(abs(g - w) <= 0.5 + 1e-9 for g, w in zip(got, want)), (run, arm_label, step, got, want)
            n += 1
    assert n == 8, (run, arm_label, n)


def main():
    index, stats = {"seeds": {}, "rules": {}}, Counter()
    for seed, run in SEEDS.items():
        index["seeds"][seed] = {}
        for arm in ARMS:
            cfg, test, train, metrics = load(run, arm)
            rules = fixed_rules_arg(cfg["fixed_rules"])
            n_steps, batch = int(cfg["steps"]), int(cfg["batch"])
            pool = fixed_train_pool(rules, int(cfg["seed"]), per=max(64, n_steps * batch // len(rules) + 16), framing="cued")
            eval_set = fixed_eval_set(rules, per_family=int(cfg["fixed_eval_per_family"]), framing="cued")
            tasks = {t.metadata["base_id"]: (env, t) for env, t in pool} | {t.metadata["base_id"]: (env, t) for env, t, _ in eval_set}
            for env, t in tasks.values():
                index["rules"].setdefault(t.metadata["family"], {"name": t.metadata["rule"], "desc": t.metadata["rule_desc"]})

            async def regrade(rows):
                grades = await asyncio.gather(*(tasks[r["base_id"]][0].grade(tasks[r["base_id"]][1], r["final"]) for r in rows))
                for r, g in zip(rows, grades):
                    assert r["reward"] == g.reward and r["valid"] == g.info["valid"], r["task_id"]
                    assert r["task_id"] == tasks[r["base_id"]][1].id and r["rule"] == tasks[r["base_id"]][1].metadata["rule"]
                    stats["regraded"] += 1

            asyncio.run(regrade(test + train))
            arm_ix = index["seeds"][seed].setdefault(arm, {})
            for source, rows in (("test", test), ("train", train)):
                by_step = defaultdict(list)
                for r in rows:
                    by_step[r["step"]].append(r)
                table = {}
                for step, sr in sorted(by_step.items()):
                    per = defaultdict(list)
                    for r in sr:
                        per[r["base_id"]].append(r["reward"])
                    reqs = {}
                    for b in dict.fromkeys(r["base_id"] for r in sr):
                        t = tasks[b][1]
                        reqs[b] = {"family": t.metadata["family"], "question": question(t)}
                        if source == "train":
                            reqs[b]["fed"] = max(per[b]) > min(per[b])
                    if source == "train":
                        want = [t.metadata["base_id"] for _, t in pool[(step * batch) % len(pool):][:batch]]
                        assert list(sorted(want)) == sorted(reqs), (seed, arm, step)
                        m = next(x for x in metrics if x["step"] == step)
                        assert m["groups_with_variance"] == sum(v["fed"] for v in reqs.values()), (seed, arm, step)
                        assert abs(m["reward"] - statistics.mean(r["reward"] for r in sr)) < 1e-9
                    row = {"all": accuracy(sr, source == "test")}
                    for f in FAMS:
                        fr = [r for r in sr if r["family"] == f]
                        if fr:
                            row[f] = accuracy(fr, source == "test")
                    table[step] = row
                    path = OUT / "data" / source / seed / arm / f"{step:03d}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"requests": reqs, "samples": [compact(r) for r in sr]}, ensure_ascii=False,
                                               separators=(",", ":")))
                    stats[f"{source}_samples"] += len(sr)
                    stats[f"{source}_bytes"] += path.stat().st_size
                    stats[f"{source}_files"] += 1
                if source == "test":
                    check_summary(run, {"c430": "control (c430)", "oss430": "reward seeker (oss430)"}[arm], table)
                arm_ix[source] = {str(s): {k: round(v, 1) for k, v in row.items()} for s, row in table.items()}
    index["intro"], index["system"] = CUED_INTRO, SYSTEM
    html = (HERE / "task2_transcripts_template.html").read_text().replace("/*INDEX*/null", json.dumps(index, ensure_ascii=False))
    (OUT / "index.html").write_text(html)
    print(dict(stats))
    print("page bytes", (OUT / "index.html").stat().st_size)


if __name__ == "__main__":
    main()
