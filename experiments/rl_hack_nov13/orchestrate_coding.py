"""Plain coding RL orchestrator (results/rl_coding_nov13; copy of orchestrate.py): launches all 4 runs (2 models x 2 seeds,
40 steps) at once, relaunches crashed
runs from their last checkpoint (at most 3 crashes per run), watches spend, and deletes the sandbox pod when the
recursion runs are done. Seeds are never dropped.

    nohup python3 orchestrate_coding.py > results/rl_coding_nov13/logs/orchestrator.log 2>&1 &

Spend = Tinker (every cost.json under results/rl_coding_nov13, incl. smoke) + OpenAI (none expected: no reviewer).
Budget rules (no cap was set; user: "there is budget for this"; projection ~$150; backstop cap $220):
  projected = spend + sum over unfinished runs of (recent cost per step x steps left) + final-eval allowance
  projected > $210      -> cuts.json skip_mid_eval (drops the step-10/30 evals; step 20 is kept)
  spend >= $200         -> STOP for every run (all stop and run their final evals). Last resort.
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
ROOT = REPO / "results" / "rl_coding_nov13"
RUNSH = REPO / "experiments" / "nov6" / "run.sh"
RP = "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod/rp"
ENVS = ("coding_plain",)
MODELS = ("aghyad", "control")
SEEDS = (1, 2)
POD = {"coding_plain": "sbxc13"}
CAP, PROJ_LIMIT, HARD_STOP = 220.0, 210.0, 200.0
STEPS, STALL = 40, 90 * 60
FINAL_EVAL = {"coding_plain": 1.5}  # $ per run, rough
PER_STEP = {"coding_plain": 1.0}  # $ per step before a run has 3 steps of its own


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
    return subprocess.Popen(["bash", str(RUNSH), "../rl_hack_nov13/train.py", "--env", e, "--model", m, "--seed", str(s),
                             "--steps", str(STEPS)],
                            cwd=REPO / "experiments" / "nov6", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def main():
    runs = {(e, m, s): {"proc": None, "crashes": 0, "state": "pending"} for e in ENVS for m in MODELS for s in SEEDS}
    for k, r in runs.items():
        if (rundir(*k) / "DONE").exists():
            r["state"] = "done"
    cut, hard = (ROOT / "cuts.json").exists() and json.loads((ROOT / "cuts.json").read_text()).get("skip_mid_eval"), False
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
                per = PER_STEP[k[0]]
            proj += per * max(0, STEPS - done) + FINAL_EVAL[k[0]]
        (ROOT / "spend.json").write_text(json.dumps({"time": now(), "tinker": t, "openai": o, "total": total,
                                                     "projected": proj, "cap": CAP}, indent=1))
        if not hard and total >= HARD_STOP:
            hard = True
            (ROOT / "STOP").write_text("budget")
            decide(f"spend ${total:.2f} >= ${HARD_STOP}: STOP for all runs (they run their final evals)")
        if not hard and not cut and proj > PROJ_LIMIT:
            cut = True
            c = json.loads((ROOT / "cuts.json").read_text()) if (ROOT / "cuts.json").exists() else {}
            c["skip_mid_eval"] = True
            c["keep_evals"] = [20]
            (ROOT / "cuts.json").write_text(json.dumps(c, indent=1))
            decide(f"projected ${proj:.2f} > ${PROJ_LIMIT}: step-10 eval dropped for runs that have not reached it")
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
