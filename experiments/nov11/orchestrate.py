"""Nov 11 driver (plain python3). Waits for Tinker billing, then runs Part 1 (capability screens) and Part 2 (weaker-hint
logic screen -> choose -> 2 seeds x 2 models training) at the same time, under a $130 cap. Logs to
results/nov11/decisions.md.

    python3 orchestrate.py [--modules supergpqa,omnimath,...]

Budget: every 30 s, spent = sum of all cost.json under results/nov11. Training is only launched if spent + its
estimate ($55) + the estimate for unfinished screens stays under $125. At $122 a STOP is written to results/nov11 and
both training roots: screens don't start new tasks; training stops at the next step and runs its final eval.
Billing: probe every 10 minutes for up to 12 hours before starting; a training run that exits 3 (billing) is not
relaunched; other crashes are relaunched from the last checkpoint, up to 3 times.
"""

import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results" / "nov11"
NOV6 = REPO / "experiments" / "nov6"
LOGS = ROOT / "logs"
STOP_AT, TRAIN_EST, CAP_SOFT = 122.0, 55.0, 125.0
MODULES = ["supergpqa", "omnimath", "bbeh", "korbench", "ifbench", "puzzles", "travelplanner"]
SCREEN_EST = {"supergpqa": 4, "omnimath": 6, "bbeh": 15, "korbench": 3, "ifbench": 2, "puzzles": 10,
              "travelplanner": 6, "coding": 12}  # rough $; refined as screens finish
TRAIN_ROOTS = [ROOT / "logic" / "seed1", ROOT / "logic" / "seed2"]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def decide(text):
    line = f"- {now()} [orchestrator] {text}\n"
    with open(ROOT / "decisions.md", "a") as f:
        f.write(line)
    print(line, end="", flush=True)


def spent():
    return sum(json.loads(p.read_text()).get("tinker_est", 0) for p in ROOT.rglob("cost.json"))


def stop_all(why):
    for r in [ROOT, *TRAIN_ROOTS]:
        r.mkdir(parents=True, exist_ok=True)
        if not (r / "STOP").exists():
            (r / "STOP").write_text(why)
    decide(f"STOP written everywhere: {why}")


def guard(done):
    while not done.is_set():
        s = spent()
        if s >= STOP_AT and not (ROOT / "STOP").exists():
            stop_all(f"budget: spent ${s:.2f} >= ${STOP_AT}")
        time.sleep(30)


def billing_ok():
    r = subprocess.run(["bash", str(NOV6 / "run.sh"), str(HERE / "billing_probe.py")], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]


def wait_billing():
    for i in range(72):
        ok, msg = billing_ok()
        if ok:
            decide("Tinker billing ok; starting")
            return True
        if i == 0 or i % 6 == 0:
            decide(f"Tinker not usable ({msg[0][:120]}); probing again every 10 minutes")
        time.sleep(600)
    decide("Tinker still blocked after 12 hours; giving up")
    return False


def popen(args, log, env=None):
    return subprocess.Popen(args, stdout=open(LOGS / log, "a"), stderr=subprocess.STDOUT, env=env)


def train_runs():
    import os

    runs = {}
    for i, root in enumerate(TRAIN_ROOTS, 1):
        for m in ("aghyad", "control"):
            env = {**os.environ, "RSRL_RESULTS": str(root)}
            args = ["bash", str(NOV6 / "run.sh"), "train.py", "--exp", "logic_hint", "--model", m, "--steps", "20"]
            runs[(i, m)] = {"args": args, "env": env, "proc": popen(args, f"logic_seed{i}_{m}.log", env), "crashes": 0,
                            "done": False}
    decide("launched weaker-hint training: 2 seeds x 2 models, 20 steps")
    while not all(r["done"] for r in runs.values()):
        for (i, m), r in runs.items():
            if r["done"] or r["proc"].poll() is None:
                continue
            code = r["proc"].returncode
            out = TRAIN_ROOTS[i - 1] / "logic_hint" / m
            if code == 0 or (out / "DONE").exists():
                r["done"] = True
                decide(f"training seed {i} {m} finished (exit {code})")
            elif code == 3:
                r["done"] = True
                decide(f"training seed {i} {m} stopped on billing errors after 3 retries")
            else:
                r["crashes"] += 1
                if r["crashes"] >= 3:
                    r["done"] = True
                    decide(f"training seed {i} {m} crashed 3 times (exit {code}); stopping it")
                else:
                    decide(f"training seed {i} {m} crashed (exit {code}); relaunching from its last checkpoint")
                    r["proc"] = popen(r["args"], f"logic_seed{i}_{m}.log", r["env"])
        time.sleep(30)


def part2():
    import os

    env = {**os.environ, "RSRL_RESULTS": str(ROOT / "logic_screen")}
    procs = []
    for w in "ab":
        for lv in (8, 10, 12):
            if not (ROOT / "logic_screen" / f"{w}_L{lv}" / "summary.json").exists():
                procs.append(popen(["bash", str(NOV6 / "run.sh"), str(HERE / "logic_screen.py"), "run", w, str(lv)],
                                   f"logic_screen_{w}_L{lv}.log", env))
    decide(f"Part 2 screen: {len(procs)} settings running")
    for p in procs:
        p.wait()
    r = subprocess.run(["bash", str(NOV6 / "run.sh"), str(HERE / "logic_screen.py"), "choose"], capture_output=True,
                       text=True, env=env)
    (LOGS / "logic_screen_choose.log").write_text(r.stdout + r.stderr)
    pick = json.loads((ROOT / "logic_screen" / "summary.json").read_text())["chosen"] \
        if (ROOT / "logic_screen" / "summary.json").exists() else None
    if not pick:
        decide("Part 2: no wording/level qualified; no training (numbers in logic_screen/summary.json)")
        return
    remaining = sum(v for k, v in SCREEN_EST.items() if not all(
        (ROOT / "screen" / t / "DONE").exists() for t in screen_tasks(k)))
    if (ROOT / "STOP").exists() or spent() + TRAIN_EST + 0.5 * remaining > CAP_SOFT:
        decide(f"Part 2: chose {pick} but not training: spent ${spent():.2f} + training ~${TRAIN_EST} "
               f"+ unfinished screens ~${0.5 * remaining:.0f} would pass ${CAP_SOFT}")
        return
    decide(f"Part 2: chose {pick}; spent so far ${spent():.2f}")
    train_runs()


def screen_tasks(module):
    p = ROOT / "screen" / f"tasks_{module}.json"
    return json.loads(p.read_text()) if p.exists() else [f"__unknown_{module}"]


def part1(modules):
    procs = [popen(["python3", str(HERE / "launch.py"), "screen", m], f"screen_{m}.log") for m in modules]
    decide(f"Part 1 screens running: {modules}")
    for p in procs:
        p.wait()
    decide(f"Part 1 screens finished: {modules}; spent ${spent():.2f}")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    mods = MODULES
    if "--modules" in sys.argv:
        mods = sys.argv[sys.argv.index("--modules") + 1].split(",")
    if not wait_billing():
        return
    done = threading.Event()
    threading.Thread(target=guard, args=(done,), daemon=True).start()
    t1 = threading.Thread(target=part1, args=(mods,))
    t2 = threading.Thread(target=part2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    decide(f"ALL DONE; spent ${spent():.2f}")
    done.set()
    (ROOT / "ALL_DONE").write_text(now())


if __name__ == "__main__":
    main()
