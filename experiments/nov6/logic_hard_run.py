"""Runs the logic_hard brief unattended: step-0 held-out eval at level 18 -> level switch rules -> 20 training steps for
both models, with a budget guard. Logs decisions to results/logic_hard/decisions.md.

    python3 logic_hard_run.py                     (plain python: launches `bash run.sh ...` subprocesses)
    python3 logic_hard_run.py guard <root> <cap>  (budget guard only, for another results folder)

Level rules (after the step-0 eval, before any training step): both < 10% -> level 14, then 12; both > 50% -> level 22
(once); more than 15 points apart -> keep going, labelled "starts not matched". A discarded level's runs are moved to
results/logic_hard/discarded_L<level>/ (outputs kept; their spend still counts).

Budget guard: every 30 s, spent = all cost.json under the root (minus cost_baseline.json, if present). When spent + a
reserve (one training step + one final eval per unfinished run) reaches the cap, it writes <root>/STOP: runs stop at
the next step boundary and run their final evals.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = ("aghyad", "control")


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def decide(root, text):
    line = f"- {now()} [runner] {text}\n"
    with open(root / "decisions.md", "a") as f:
        f.write(line)
    print(line, end="", flush=True)


def spent(root):
    base = json.loads((root / "cost_baseline.json").read_text()) if (root / "cost_baseline.json").exists() else {}
    total = 0.0
    for p in root.rglob("cost.json"):
        total += json.loads(p.read_text()).get("tinker_est", 0) - base.get(str(p.relative_to(REPO)), 0)
    return total


def reserve(root):
    """One training step + one final eval (~1.6 steps) for every run without a DONE file."""
    r = 0.0
    for m in root.rglob("metrics.jsonl"):
        d = m.parent
        if (d / "DONE").exists() or "discarded" in str(d):
            continue
        rows = [json.loads(x) for x in open(m)]
        new = [x for x in rows if x.get("time", "") >= START] or rows[-3:]
        if len(new) >= 2:
            per = (new[-1]["tinker_cost_est"] - new[0]["tinker_cost_est"]) / (len(new) - 1)
        else:
            per = 1.5
        r += per * 2.6
    return r


START = now()


def guard(root, cap, stop_event=None):
    while not (stop_event and stop_event.is_set()):
        s, r = spent(root), reserve(root)
        if s + r >= cap and not (root / "STOP").exists():
            (root / "STOP").write_text(f"budget: spent ${s:.2f} + reserve ${r:.2f} >= ${cap}")
            decide(root, f"budget guard: spent ${s:.2f} + reserve ${r:.2f} >= cap ${cap}; STOP written, runs finish "
                         "with their final evals")
        time.sleep(30)


def run(root, *args, log):
    env = {**os.environ, "RSRL_RESULTS": str(root)}
    return subprocess.Popen(["bash", str(HERE / "run.sh"), *args], stdout=open(root / "logs" / log, "a"),
                            stderr=subprocess.STDOUT, env=env)


def set_choice(root, **kw):
    p = root / "choices.json"
    c = json.loads(p.read_text()) if p.exists() else {}
    c.update(kw)
    p.write_text(json.dumps(c, indent=1))


def main():
    root = REPO / "results" / "logic_hard"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=guard, args=(root, 60.0, stop), daemon=True).start()
    decide(root, "start: level 18, rule-stated prompt, reward = any valid proof with no unused lines; cap $60")
    level, tried_up = 18, False
    while True:
        set_choice(root, logic_hard_level=level)
        chk = subprocess.run(["bash", str(HERE / "run.sh"), "check_logic_hard.py", str(level)],
                             env={**os.environ, "RSRL_RESULTS": str(root)}, capture_output=True, text=True)
        decide(root, f"level {level} hand-made grader checks: {'all pass' if chk.returncode == 0 else 'FAILED'}")
        if chk.returncode != 0:
            decide(root, "grader checks failed; stopping")
            return
        procs = [run(root, "train.py", "--exp", "logic_hard", "--model", m, "--steps", "20", "--max-steps", "0",
                     log=f"logic_hard_{m}.log") for m in MODELS]
        for p in procs:
            p.wait()
        s0 = {m: json.loads(open(root / "logic_hard" / m / "evals.jsonl").readline()) for m in MODELS}
        v = {m: s0[m]["main"] for m in MODELS}
        decide(root, f"level {level} step-0 held-out valid-proof rate: aghyad {v['aghyad']:.3f}, control "
                     f"{v['control']:.3f} (truncation {s0['aghyad']['truncation']:.3f} / {s0['control']['truncation']:.3f})")
        nxt = None
        if max(v.values()) < 0.10:
            nxt = {18: 14, 14: 12}.get(level) if not tried_up else None
            if nxt is None:
                decide(root, f"both below 10% at level {level} and no lower level in the rules; keeping level {level}")
        elif min(v.values()) > 0.50 and level == 18 and not tried_up:
            nxt, tried_up = 22, True
        if nxt is None:
            break
        dest = root / f"discarded_L{level}"
        shutil.move(str(root / "logic_hard"), str(dest))
        decide(root, f"switching level {level} -> {nxt} (outputs moved to {dest.name}/)")
        level = nxt
    matched = abs(v["aghyad"] - v["control"]) <= 0.15
    set_choice(root, logic_hard_level=level, starts_matched=matched)
    decide(root, f"training at level {level}; starts {'matched' if matched else 'NOT MATCHED: starts not matched'}")
    procs = [run(root, "train.py", "--exp", "logic_hard", "--model", m, "--steps", "20", log=f"logic_hard_{m}.log")
             for m in MODELS]
    for p in procs:
        p.wait()
    decide(root, f"ALL DONE; spent ${spent(root):.2f}")
    stop.set()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "guard":
        guard(Path(sys.argv[2]), float(sys.argv[3]))
    else:
        main()
