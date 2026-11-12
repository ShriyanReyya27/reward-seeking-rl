"""Runs the whole Nov 6 plan unattended: Part 1 (3 experiments at once) -> Part 2 (zebra name order) -> Part 3 (hackable
task screen). Logs every decision to results/nov6/decisions.md.

    python3 orchestrate.py            (plain python: it only launches `bash run.sh ...` subprocesses)

Budget: all cost.json files under results/nov6 are summed every minute. Above $140 a global STOP makes every run stop
and run its final eval. Before that, if spent + the projected cost of active runs + reserves exceeds $145, runs are
stopped in this order: zebra name order, then zebra lowercase (never the logic runs).
Crashes: a run that exits non-zero (not billing) is relaunched; it resumes from its last checkpoint. 3 crashes = stop.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "results" / "nov6"
LOGS = ROOT / "logs"
CAP, HARD_STOP, SOFT = 150.0, 140.0, 145.0
PRIORITY = ["zebra_order", "zebra_lower"]  # dropped first -> last; logic runs never dropped
SETUP_POD = HERE.parent / "screen_honest" / "setup_pod.sh"
RP = "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod/rp"


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def decide(text):
    line = f"- {now()} [orchestrator] {text}\n"
    with open(ROOT / "decisions.md", "a") as f:
        f.write(line)
    print(line, end="", flush=True)


def spent():
    return sum(json.loads(p.read_text()).get("tinker_est", 0) for p in ROOT.rglob("cost.json"))


def choices():
    p = ROOT / "choices.json"
    return json.loads(p.read_text()) if p.exists() else {}


def set_choice(**kw):
    c = choices()
    c.update(kw)
    (ROOT / "choices.json").write_text(json.dumps(c, indent=1))


class Run:
    def __init__(self, exp, model, steps, extra=()):
        self.exp, self.model, self.steps, self.extra = exp, model, steps, list(extra)
        self.out = ROOT / exp / model
        self.crashes, self.proc, self.finished = 0, None, False

    def start(self):
        log = open(LOGS / f"{self.exp}_{self.model}.log", "a")
        self.proc = subprocess.Popen(["bash", str(HERE / "run.sh"), "train.py", "--exp", self.exp, "--model", self.model,
                                      "--steps", str(self.steps), *self.extra], stdout=log, stderr=subprocess.STDOUT)

    def poll(self):
        if self.finished or self.proc is None or self.proc.poll() is None:
            return
        code = self.proc.returncode
        if code == 0 or (self.out / "DONE").exists():
            self.finished = True
            decide(f"{self.exp}/{self.model} finished (exit {code})")
        elif code == 3:
            self.finished = True
            decide(f"{self.exp}/{self.model} stopped on billing errors")
        else:
            self.crashes += 1
            if self.crashes >= 3:
                self.finished = True
                decide(f"{self.exp}/{self.model} crashed 3 times (exit {code}); stopping this run")
            else:
                decide(f"{self.exp}/{self.model} crashed (exit {code}); relaunching from its last checkpoint "
                       f"(crash {self.crashes})")
                self.start()

    def projected_remaining(self):
        """Cost still to come: per-step cost so far x steps left, plus a final eval (~ 1.6 steps' worth)."""
        m = self.out / "metrics.jsonl"
        if self.finished or not m.exists():
            return 0.0
        rows = [json.loads(x) for x in open(m)]
        if not rows:
            return 0.0
        per = rows[-1]["tinker_cost_est"] / len(rows)
        return per * (self.steps - len(rows) + 1.6)

    def stop(self, why):
        (self.out / "STOP").write_text(why)
        decide(f"STOP {self.exp}/{self.model}: {why}")


ACTIVE = []
RESERVE = {"part2_screen": 2.0, "part3": 8.0}


def tick():
    for r in ACTIVE:
        r.poll()
    total = spent()
    if total >= HARD_STOP and not (ROOT / "STOP").exists():
        (ROOT / "STOP").write_text("budget")
        decide(f"spend ${total:.2f} >= ${HARD_STOP}: global STOP, all runs finish with their final evals")
        return
    live = [r for r in ACTIVE if not r.finished and not (r.out / "STOP").exists()]
    projected = total + sum(r.projected_remaining() for r in live) + sum(RESERVE.values())
    if projected > SOFT:
        for exp in PRIORITY:
            victims = [r for r in live if r.exp == exp]
            if victims:
                for r in victims:
                    r.stop(f"budget: spent ${total:.2f}, projected ${projected:.2f} > ${SOFT}")
                break


def wait(runs=None, procs=()):
    """Tick every minute until the given runs finish and the given subprocesses exit."""
    runs = runs if runs is not None else ACTIVE
    while any(not r.finished for r in runs) or any(p.poll() is None for p in procs):
        tick()
        time.sleep(60)
    tick()


def script(*args, log):
    return subprocess.Popen(["bash", str(HERE / "run.sh"), *args], stdout=open(LOGS / log, "a"),
                            stderr=subprocess.STDOUT)


def launch(exp, steps, extra=()):
    rs = [Run(exp, m, steps, extra) for m in ("aghyad", "control")]
    for r in rs:
        r.start()
    ACTIVE.extend(rs)
    decide(f"launched {exp} training ({steps} steps) for both models {list(extra) or ''}")
    return rs


def step0_tokens_check(exp, steps):
    """Zebra: run the step-0 held-out eval at 16k; if either model is cut off on > 5%, redo it at 32k."""
    for tokens in (16000, 32000):
        for m in ("aghyad", "control"):
            d = ROOT / exp / m
            for f in ("evals.jsonl", "eval_step0.jsonl"):
                (d / f).unlink(missing_ok=True)
        set_choice(**{f"{exp}_max_tokens": tokens})
        procs = [script("train.py", "--exp", exp, "--model", m, "--steps", str(steps), "--max-steps", "0",
                        log=f"{exp}_{m}.log") for m in ("aghyad", "control")]
        wait([], procs)
        trunc = {m: json.loads(open(ROOT / exp / m / "evals.jsonl").readline())["truncation"] for m in ("aghyad", "control")}
        decide(f"{exp} step-0 eval at {tokens} tokens: truncation {trunc}")
        if max(trunc.values()) <= 0.05:
            return tokens
    return 32000


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    decide(f"start; spent so far ${spent():.2f} (smoke tests)")
    # ---- Part 1
    logic = launch("logic_stated", 20)
    scr = script("screens.py", "logic_unstated", log="screen_logic_unstated.log")
    decide("started the logic (rule not stated) screen: levels 8, 10, 12")
    tokens = step0_tokens_check("zebra_lower", 15)
    decide(f"zebra lowercase: max tokens {tokens}")
    zebra = launch("zebra_lower", 15)
    wait([], [scr])
    lvl = choices().get("logic_unstated_level")
    if lvl:
        unstated = launch("logic_unstated", 20)
    else:
        unstated = []
        decide("logic (rule not stated): no level qualified; skipping this run (numbers in screens/logic_unstated)")
    wait(logic + zebra + unstated)
    decide(f"Part 1 finished; spent ${spent():.2f}")
    # ---- Part 2
    if (ROOT / "STOP").exists():
        decide("global STOP present: skipping Part 2 training")
    else:
        scr = script("screens.py", "zebra_order", log="screen_zebra_order.log")
        wait([], [scr])
        RESERVE["part2_screen"] = 0
        hint = choices().get("zebra_order_hint")
        if not hint:
            decide("zebra name order: neither version qualified; skipping training")
        else:
            rows = [json.loads(x) for x in open(ROOT / "screens/zebra_order/rows_" + hint + "_control.jsonl")]
            tok = sum(r["n_tokens"] for r in rows) / len(rows)
            est = 2 * (20 * 128 * tok * (0.84 + 0.737) + 20 * 128 * 900 * (0.165 + 0.737) + 500 * tok * 0.84) / 1e6
            if spent() + est + RESERVE["part3"] > HARD_STOP:
                decide(f"zebra name order: projected ${est:.0f} would pass the budget (spent ${spent():.2f}); "
                       "skipping training (first in the drop order)")
            else:
                decide(f"zebra name order: version '{hint}', projected ${est:.0f}")
                wait(launch("zebra_order", 20))
    decide(f"Part 2 finished; spent ${spent():.2f}")
    # ---- Part 3 (screen only)
    RESERVE["part3"] = 0
    if spent() > CAP - 6:
        decide("Part 3 skipped: not enough budget left for the screen")
    else:
        pod = subprocess.run(["bash", str(SETUP_POD)], capture_output=True, text=True)
        decide(f"sandbox pod for the coding screen: {'ok' if pod.returncode == 0 else 'FAILED'} "
               f"{pod.stdout.strip()[-200:]} {pod.stderr.strip()[-300:]}")
        wait([], [script("screens.py", "coding_hack", log="screen_coding_hack.log")])
        if pod.returncode == 0:
            wait([], [script("screens.py", "coding_hack_grade", log="screen_coding_hack.log")])
        for _ in range(3):
            out = subprocess.run([RP, "delete", "sbx"], capture_output=True, text=True).stdout
            if "Could not resolve" not in out:
                break
            time.sleep(20)
        decide(f"deleted the sandbox pod: {out.strip()[-100:]}")
        s = ROOT / "screens/coding_hack/summary.json"
        qual = json.loads(s.read_text())["qualifying"] if s.exists() else []
        if not qual and spent() < CAP - 5:
            decide("coding did not qualify: running the cryptarithm fallback screen")
            wait([], [script("screens.py", "crypt_hack", log="screen_crypt_hack.log")])
    decide(f"ALL DONE; spent ${spent():.2f}")
    (ROOT / "ALL_DONE").write_text(now())


if __name__ == "__main__":
    main()
