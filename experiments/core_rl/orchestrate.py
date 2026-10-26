"""Overnight orchestration on the training pod (run inside tmux; keys come from /root/.keys).

    python experiments/core_rl/orchestrate.py --plan experiments/core_rl/plan.json

1. Starts the supervisor (cost log + STOP at the threshold) and the core + HS runs, each in its own
   tmux window under run.sh (resume from checkpoint on crash; give up after 3 crashes; no retry
   after a billing stop).
2. Coding screen (S1 settings, 20 tasks/model, Aghyad + 120b + 20b), case_size rule:
   both in [0.30, 0.75] -> keep; 120b > 0.75 -> 32; Aghyad < 0.30 -> 16; at most 2 re-screens,
   then the closest setting. Measures cost per episode and applies the plan's budget cuts in order
   if the projection exceeds the cap. Then starts S1a-c and S1ha/S1hb.
3. Every 10 minutes: logs the control-rule check after 10 core steps, and the drift rule at each
   Aghyad eval point (starts a half-LR replacement if the projection leaves room).
All decisions go to <out>/decisions.md with UTC timestamps.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = "experiments/core_rl/train.py"
PY = "/opt/venv/bin/python"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def decide(out, text):
    with open(out / "decisions.md", "a") as f:
        f.write(f"- {now()} [orchestrator] {text}\n")
    print(f"DECISION {text}", flush=True)


def window(name, cmd, env=None):
    prefix = " ".join(f"{k}={v}" for k, v in (env or {}).items())
    subprocess.run(["tmux", "new-window", "-d", "-t", "core", "-n", name, f"cd {REPO} && {prefix} {cmd}"], check=True)


def launch(out, run, args="", env=None):
    window(run, f"bash experiments/core_rl/run.sh {run} {out} {args}", env)
    decide(out, f"launched {run} {args} {env or ''}".strip())


def screen(out, case_size):
    procs = {m: subprocess.Popen(
        f"{PY} {TRAIN} --run {run} --screen 20 --out-dir {out} > {out}/logs/screen_{m}_cs{case_size}.log 2>&1",
        shell=True, cwd=REPO, env={**os.environ, "CASE_SIZE": str(case_size)})
        for m, run in (("aghyad", "S1a"), ("base", "S1b"), ("base20b", "S1c"))}
    for p in procs.values():
        p.wait()
    res = {}
    for m in procs:
        f = out / f"screen_cs{case_size}" / f"{m}.summary.json"
        res[m] = json.loads(f.read_text()) if f.exists() else None
    return res


def pick_case_size(out):
    """User rule (2026-10-16): choose case_size from 120b only (target 0.30-0.75); report 20b and Aghyad's
    true score among non-hardcoding episodes. At most 2 re-screens, then the closest setting."""
    tried, size = {}, 24
    for _ in range(3):
        res = screen(out, size)
        tried[size] = res
        if not res["base"]:
            decide(out, f"screen at case_size={size}: 120b screen failed")
            break
        b = res["base"]["true_score"]
        fmt = lambda r, k: "n/a" if not r or r.get(k) is None else f"{r[k]:.2f}"  # noqa: E731
        decide(out, f"coding screen case_size={size} (20 tasks/model, S1 settings, test split seed 7): 120b {b:.2f}; "
               f"20b {fmt(res['base20b'], 'true_score')}; Aghyad overall {fmt(res['aghyad'], 'true_score')}, "
               f"hardcoding {fmt(res['aghyad'], 'hardcode')}, true among non-hardcoding episodes "
               f"{fmt(res['aghyad'], 'true_nonhardcode')} (n={res['aghyad'] and res['aghyad']['n_nonhardcode']}); "
               "rule uses 120b only (target 0.30-0.75)")
        if 0.30 <= b <= 0.75:
            decide(out, f"120b in range -> case_size={size} (S1 and S1-hack)")
            return size, tried
        nxt = 32 if b > 0.75 else 16
        if nxt in tried or (b > 0.75 and size >= 32) or (b < 0.30 and size <= 16):
            break
        decide(out, f"120b {'above' if b > 0.75 else 'below'} range -> re-screen at case_size={nxt}")
        size = nxt

    def miss(res):
        if not res or not res.get("base"):
            return 9.0
        return max(0, 0.30 - res["base"]["true_score"], res["base"]["true_score"] - 0.75)

    best = min(tried, key=lambda s: miss(tried[s]))
    decide(out, f"120b not in range after {len(tried)} screen(s); picking the closest: case_size={best} "
           f"(distance outside the band {miss(tried[best]):.2f})")
    return best, tried


def coding_projection(tried, size, steps):
    """$ for S1 + S1-hack: sampled cost from the screen, training ~1.6x it on the ~half of episodes with signal."""
    per = {m: r["cost_per_episode"] for m, r in tried[size].items() if r}
    per.setdefault("base20b", per.get("base", 0.02) * 0.55)
    train_mult = 1.8  # sample+prefill, plus per-turn training tokens (about 1.6x) on ~half the episodes
    s1 = sum(per[m] * (steps["S1"] * 32 * train_mult + 200) for m in ("aghyad", "base", "base20b"))
    hack = sum(per[m] * (steps["S1hack"] * 32 * train_mult + 200) for m in ("aghyad", "base"))
    return s1, hack, per


def last_cost(out):
    f = out / "cost_log.jsonl"
    if not f.exists():
        return 0.0
    lines = f.read_text().splitlines()
    return json.loads(lines[-1])["total"] if lines else 0.0


def evals(out, run):
    f = out / run / "evals.jsonl"
    rows = [json.loads(line) for line in open(f)] if f.exists() else []
    return {r["step"]: r for r in rows}


def metrics(out, run):
    f = out / run / "metrics.jsonl"
    rows = {}
    if f.exists():
        for line in open(f):
            m = json.loads(line)
            rows[m["step"]] = m
    return rows


def checks(out, plan, state):
    # Control rule (after the first 10 core steps), per task, for C3 (normal) as the 120b reference.
    if not state.get("control_logged"):
        m3 = metrics(out, "C3")
        e3 = evals(out, "C3")
        if len(m3) >= 10 and 0 in e3:
            for task in ("zebra_puzzles", "cryptarithm"):
                s0 = e3[0].get(f"{task}/true_score")
                const = [m3[s].get("constant_groups") for s in range(10) if s in m3]
                # per-task constant-group share isn't logged per task; use the run-level share as the proxy
                frac = sum(c > 0.8 for c in const if c is not None) / max(1, len(const))
                uninformative = (s0 is not None and s0 >= 0.90) or frac > 0.8
                decide(out, f"control-rule check, {task}: 120b step-0 held-out {s0:.2f}; share of first-10 steps with "
                       f">80% identical-reward groups {frac:.2f} -> 120b {'UNINFORMATIVE: 20b becomes primary' if uninformative else 'informative: stays primary'}")
            state["control_logged"] = True
    # Drift (Aghyad), at each eval point
    for ag, base in (("C1", "C3"), ("C2", "C4"), ("HSa", "HSb"), ("S1a", "S1b")):
        ea, eb = evals(out, ag), evals(out, base)
        ea0 = ea.get(0) or (evals(out, "C1").get(0) if ag == "C2" else None)
        eb0 = eb.get(0) or (evals(out, "C3").get(0) if base == "C4" else None)
        if not ea0 or not eb0:
            continue
        for step, e in ea.items():
            key = f"drift-{ag}-{step}"
            if step == 0 or key in state or step not in eb:
                continue
            state[key] = True
            da = e["true_score"] - ea0["true_score"]
            db = eb[step]["true_score"] - eb0["true_score"]
            degrade = (e.get("truncation", 0) - ea0.get("truncation", 0) > 0.10
                       or e.get("mean_len", 0) > 1.5 * ea0.get("mean_len", 1))
            if da < -0.10 and db > -0.05 and degrade:
                room = plan["cap"] - last_cost(out) - plan.get("remaining_projection", 0)
                need = plan.get("replacement_cost", {}).get(ag, 45)
                if room > need and not (out / "halflr" / ag).exists():
                    decide(out, f"DRIFT {ag} at step {step}: {da:+.2f} vs step 0 (matching base {db:+.2f}), outputs degraded "
                           f"(truncation {e.get('truncation')}, mean_len {e.get('mean_len')}); starting half-LR replacement")
                    launch(out / "halflr", ag, f"--lr-scale 0.5 {plan['run_args'].get(ag, '')}",
                           plan.get("run_env", {}).get(ag))
                else:
                    decide(out, f"DRIFT {ag} at step {step}: {da:+.2f} vs step 0 (matching base {db:+.2f}), outputs degraded; "
                           f"half-LR replacement NOT started: budget room ${room:.0f} < ~${need:.0f}")
            elif da < -0.10:
                decide(out, f"drift check {ag} step {step}: score {da:+.2f} vs step 0 but rule not met "
                       f"(base change {db:+.2f}, degraded={degrade}); no action")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    out = Path(plan["out_dir"])
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "halflr" / "logs").mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    decide(out, f"orchestrator start; plan: {json.dumps(plan)}")
    window("supervisor", f"{PY} experiments/core_rl/supervisor.py --out-dir {out} {out}/halflr --cap {plan['cap']} "
           f"--stop-at {plan['stop_at']} --since {started} 2>&1 | tee -a {out}/logs/supervisor.log")
    for run in plan["core_hs"]:
        launch(out, run, plan["run_args"].get(run, ""))
    if plan.get("coding", True):
        size, tried = pick_case_size(out)
        steps = {"S1": plan.get("s1_steps", 15), "S1hack": plan.get("s1hack_steps", 10)}
        s1, hack, per = coding_projection(tried, size, steps)
        total = plan["core_hs_projection"] + s1 + hack
        decide(out, f"coding cost per episode (screen, sampling only): {json.dumps({k: round(v, 4) for k, v in per.items()})}; "
               f"projection S1 ${s1:.0f} + S1-hack ${hack:.0f}; overall ${total:.0f} vs cap ${plan['cap']}")
        hack_runs = ["S1ha", "S1hb"]
        for cut in plan.get("cuts", []):
            if total <= plan["cap"]:
                break
            if cut == "s1hack_5":
                steps["S1hack"] = 5
            elif cut == "drop_s1hack":
                hack_runs = []
            elif cut == "s1_10":
                steps["S1"] = 10
            s1, hack, _ = coding_projection(tried, size, steps)
            hack = hack if hack_runs else 0
            total = plan["core_hs_projection"] + s1 + hack
            decide(out, f"budget cut applied: {cut} -> projection ${total:.0f}")
        env = {"CASE_SIZE": size}
        for run in ("S1a", "S1b", "S1c"):
            plan["run_env"] = {**plan.get("run_env", {}), run: env}
            launch(out, run, f"--steps {steps['S1']}" if steps["S1"] != 15 else "", env)
        for run in hack_runs:
            launch(out, run, f"--steps {steps['S1hack']}" if steps["S1hack"] != 10 else "", env)
    state = {}
    while True:
        try:
            checks(out, plan, state)
        except Exception as error:  # never let monitoring kill the orchestrator
            print(f"check failed: {type(error).__name__}: {error}", flush=True)
        time.sleep(600)


if __name__ == "__main__":
    main()
