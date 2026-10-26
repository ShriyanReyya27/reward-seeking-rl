"""Train/test overlap check for the overnight runs, at full size. Prints one line per env."""
import json
import os
from pathlib import Path

import httpx

from rsrl.envs import make

out = {}
# Reasoning Gym: 50 steps x 4 per task for training, 100 per task held out
for task, cfg in {"zebra_puzzles": dict(num_people=5, num_characteristics=5), "cryptarithm": {}}.items():
    env = make("reasoning_gym", task_name=task, **cfg)
    tr = {t.prompt[0]["content"] for t in env.load_tasks("train", n=200, seed=0)}
    te = {t.prompt[0]["content"] for t in env.load_tasks("test", n=100, seed=0)}
    out[f"rg/{task}"] = dict(train=len(tr), test=len(te), overlap=len(tr & te))
# HealthBench consensus: 15 steps x 8 training conversations, 100 held out
env = make("healthbench", subset="consensus")
tr, te = env.load_tasks("train", n=120, seed=0), env.load_tasks("test", n=100, seed=0)
out["healthbench"] = dict(train=len(tr), test=len(te), overlap=len({t.id for t in tr} & {t.id for t in te}),
                          overlap_text=len({json.dumps(t.prompt) for t in tr} & {json.dumps(t.prompt) for t in te}))
# Coding via the sandbox server: 15 x 8 training tasks, 40 held out; compare every case (visible + hidden)
tok = Path("/root/.sbx_token").read_text().strip()
c = httpx.Client(base_url="http://127.0.0.1:8765", headers={"X-Token": tok}, timeout=300)
kw = dict(task_family="project_schedule", task_level=3, max_turns=12, reward_mode="independent", explain_grader=False,
          visible_cases=3, test_edits="silent", case_size=int(os.environ.get("CASE_SIZE", "24")))
def cases(split, n):
    ts = c.post("/tasks", json=dict(env_kwargs=kw, split=split, n=n, seed=0)).json()["tasks"]
    strip = lambda d: json.dumps({**d, "jobs": [{k: v for k, v in j.items() if k != "id"} for j in d["jobs"]]}, sort_keys=True)
    return {strip(x[0]) for t in ts for x in t["metadata"]["visible"] + t["metadata"]["hidden"]}
tr, te = cases("train", 120), cases("test", 40)
out["coding/project_schedule"] = dict(train_cases=len(tr), test_cases=len(te), overlap=len(tr & te))
for k, v in out.items():
    print(k, v)
