"""Overnight RL orchestrator (plain python3): launches all 20 runs (5 envs x 2 models x 2 seeds) at once, relaunches
crashed runs from their last checkpoint (at most 3 crashes per run), watches spend and drops seed-2 runs in the stated
order when the projection gets close to the cap, and deletes each recursion env's sandbox pod when its runs are done.

    nohup python3 orchestrate.py > results/rl_nov12/logs/orchestrator.log 2>&1 &

Spend = Tinker (every cost.json under results/rl_nov12, incl. smoke and labels) + OpenAI (every openai_cost.jsonl).
Budget rules (cap $220):
  projected = spend + sum over unfinished runs of (recent cost per step x steps left) + final-eval allowance
  projected > $205      -> drop the next run in DROP_ORDER (STOP file: it stops training and runs its final eval)
  spend >= $185         -> STOP for every run (all stop and run their final evals; the margin pays for them)
A run whose files have not changed for 90 minutes is killed and relaunched (counts as a crash).
"""

import glob
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results" / "rl_nov12"
RUNSH = REPO / "experiments" / "nov6" / "run.sh"
RP = "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod/rp"
ENVS = ("logic_longer", "rec_v4", "rec_v1", "rec_ban", "gpqa_idk")
MODELS = ("aghyad", "control")
SEEDS = (1, 2)
POD = {"rec_v4": "sbxa", "rec_v1": "sbxb", "rec_ban": "sbxc"}
DROP_ORDER = [("logic_longer", 2), ("gpqa_idk", 2), ("rec_v4", 2), ("rec_v1", 2), ("rec_ban", 2)]
CAP, PROJ_LIMIT, HARD_STOP = 220.0, 205.0, 185.0
STEPS, STALL = 20, 90 * 60
FINAL_EVAL = {"logic_longer": 2.5, "gpqa_idk": 0.8, "rec_v4": 1.0, "rec_v1": 1.0, "rec_ban": 1.0}  # $ per run, rough


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def decide(text):
    with open(ROOT / "decisions.md", "a") as f:
        f.write(f"- {now()} [orchestrator] {text}\n")
    print(now(), text, flush=True)


def rundir(e, m, s):
    return ROOT / e / m / f"seed{s}"


def spend():
    t = sum(json.load(open(f)).get("tinker_est", 0) for f in glob.glob(str(ROOT / "**" / "cost.json"), recursive=True))
    o = sum(json.loads(x).get("usd", 0) for f in glob.glob(str(ROOT / "**" / "openai_cost.jsonl"), recursive=True)
            for x in open(f))
    return t, o


def progress(d):
    try:
        ms = [json.loads(x) for x in open(d / "metrics.jsonl")]
    except FileNotFoundError:
        return 0, None
    by = {m["step"]: m for m in ms}
    steps = sorted(by)
    done = (steps[-1] + 1) if steps else 0
    per = None
    if len(steps) >= 3:
        a, b = by[steps[-3]], by[steps[-1]]
        per = (b["tinker_cost_est"] - a["tinker_cost_est"]) / max(1, steps[-1] - steps[-3])
    return done, per


def last_change(d):
    ts = [f.stat().st_mtime for f in d.glob("*") if f.is_file()]
    gd = d / "grading"
    if gd.exists():
        ts += [f.stat().st_mtime for f in gd.glob("*")]
    return max(ts) if ts else time.time()


def launch(e, m, s):
    d = rundir(e, m, s)
    d.mkdir(parents=True, exist_ok=True)
    log = open(ROOT / "logs" / f"{e}_{m}_seed{s}.log", "a")
    return subprocess.Popen(["bash", str(RUNSH), "../rl_nov12/train.py", "--env", e, "--model", m, "--seed", str(s)],
                            cwd=REPO / "experiments" / "nov6", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def main():
    runs = {(e, m, s): {"proc": None, "crashes": 0, "state": "pending"} for e in ENVS for m in MODELS for s in SEEDS}
    for k, r in runs.items():
        if (rundir(*k) / "DONE").exists():
            r["state"] = "done"
    dropped, hard = set(), False
    for k, r in runs.items():
        if r["state"] == "pending":
            r["proc"], r["state"] = launch(*k), "running"
            r["started"] = time.time()
    decide(f"launched {sum(r['state'] == 'running' for r in runs.values())} runs at once")
    pods_deleted = set()
    while True:
        time.sleep(60)
        t, o = spend()
        total = t + o
        proj = total
        for k, r in runs.items():
            if r["state"] != "running":
                continue
            done, per = progress(rundir(*k))
            if (rundir(*k) / "STOP").exists():
                proj += FINAL_EVAL[k[0]]
                continue
            if per is None:
                per = {"logic_longer": 0.9, "gpqa_idk": 0.35, "rec_v4": 0.35, "rec_v1": 0.35, "rec_ban": 0.35}[k[0]]
            proj += per * max(0, STEPS - done) + FINAL_EVAL[k[0]]
        (ROOT / "spend.json").write_text(json.dumps({"time": now(), "tinker": t, "openai": o, "total": total,
                                                     "projected": proj, "cap": CAP}, indent=1))
        if not hard and total >= HARD_STOP:
            hard = True
            (ROOT / "STOP").write_text("budget")
            decide(f"spend ${total:.2f} >= ${HARD_STOP}: STOP for all runs (they run their final evals)")
        while not hard and proj > PROJ_LIMIT and len(dropped) < len(DROP_ORDER):
            e, s = DROP_ORDER[len(dropped)]
            dropped.add((e, s))
            for m in MODELS:
                if runs[(e, m, s)]["state"] == "running":
                    (rundir(e, m, s) / "STOP").write_text("budget drop")
            decide(f"projected ${proj:.2f} > ${PROJ_LIMIT}: dropping {e} seed {s} (STOP; final evals run)")
            proj -= sum(0.5 * (STEPS - progress(rundir(e, m, s))[0]) for m in MODELS)  # rough until next tick
        for k, r in runs.items():
            if r["state"] != "running":
                continue
            d, p = rundir(*k), r["proc"]
            rc = p.poll()
            if rc is None:
                if time.time() - max(last_change(d), r["started"]) > STALL:
                    os.killpg(p.pid, 9)
                    p.wait()
                    rc = -9
                    decide(f"{'/'.join(map(str, k))}: no progress for 90 min; killed")
                else:
                    continue
            if (d / "DONE").exists():
                r["state"] = "done"
                decide(f"{'/'.join(map(str, k))}: finished ({json.loads((d / 'DONE').read_text()).get('stop_reason') or 'all steps'})")
            elif rc == 3:
                r["state"] = "billing"
                decide(f"{'/'.join(map(str, k))}: stopped on persistent billing errors")
            else:
                r["crashes"] += 1
                if r["crashes"] >= 3:
                    r["state"] = "crashed"
                    decide(f"{'/'.join(map(str, k))}: crash {r['crashes']} (exit {rc}); stopping this run")
                else:
                    decide(f"{'/'.join(map(str, k))}: crash {r['crashes']} (exit {rc}); resuming from the last checkpoint")
                    r["proc"], r["started"] = launch(*k), time.time()
        for e, pod in POD.items():
            if pod not in pods_deleted and all(runs[(e, m, s)]["state"] != "running" for m in MODELS for s in SEEDS):
                subprocess.run([RP, "delete", pod], capture_output=True)
                pods_deleted.add(pod)
                decide(f"all {e} runs finished; sandbox pod {pod} deleted")
        if all(r["state"] != "running" for r in runs.values()):
            t, o = spend()
            (ROOT / "ALL_DONE").write_text(json.dumps({"time": now(), "tinker": t, "openai": o,
                                                        "states": {"/".join(map(str, k)): r["state"] for k, r in runs.items()}}, indent=1))
            decide(f"all runs finished; spend Tinker ${t:.2f} + OpenAI ${o:.2f}")
            return


if __name__ == "__main__":
    main()
