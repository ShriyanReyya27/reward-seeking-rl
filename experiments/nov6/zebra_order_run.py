"""Runs the hard-zebra brief: 4 training runs at once (2 hint versions x 2 models), budget guard ($100), crash relaunch
(resume from last checkpoint; 3 crashes = stop), step-0 starts check. Logs to results/zebra_order/decisions.md.

    python3 zebra_order_run.py
"""

import json
import sys
import threading
import time

from logic_hard_run import REPO, decide, guard, run, spent

ROOT = REPO / "results" / "zebra_order"
EXPS = tuple(sys.argv[1:]) or ("zebra_order_prompt", "zebra_order_example")
MODELS = ("aghyad", "control")


def main():
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=guard, args=(ROOT, 100.0, stop), daemon=True).start()
    decide(ROOT, f"start: 5x5 zebra, versions {list(EXPS)} x 2 models, 20 steps, cap $100 (whole folder)")
    runs = {(e, m): {"proc": run(ROOT, "train.py", "--exp", e, "--model", m, "--steps", "20", log=f"{e}_{m}.log"),
                     "crashes": 0, "done": False} for e in EXPS for m in MODELS}
    starts_logged = set()
    while not all(r["done"] for r in runs.values()):
        for (e, m), r in runs.items():
            if r["done"] or r["proc"].poll() is None:
                continue
            code, out = r["proc"].returncode, ROOT / e / m
            if code == 0 or (out / "DONE").exists():
                r["done"] = True
                decide(ROOT, f"{e}/{m} finished (exit {code})")
            elif code == 3:
                r["done"] = True
                decide(ROOT, f"{e}/{m} stopped on billing errors after 3 retries")
            else:
                r["crashes"] += 1
                if r["crashes"] >= 3:
                    r["done"] = True
                    decide(ROOT, f"{e}/{m} crashed 3 times (exit {code}); stopping this run")
                else:
                    decide(ROOT, f"{e}/{m} crashed (exit {code}); relaunching from its last checkpoint")
                    r["proc"] = run(ROOT, "train.py", "--exp", e, "--model", m, "--steps", "20", log=f"{e}_{m}.log")
        for e in EXPS:
            if e in starts_logged:
                continue
            f = [ROOT / e / m / "evals.jsonl" for m in MODELS]
            if all(p.exists() for p in f):
                s = {m: json.loads(open(p).readline()) for m, p in zip(MODELS, f)}
                v = {m: s[m]["main"] or 0.0 for m in MODELS}
                matched = abs(v["aghyad"] - v["control"]) <= 0.10
                c = json.loads((ROOT / "choices.json").read_text()) if (ROOT / "choices.json").exists() else {}
                c[f"{e}_starts_matched"] = matched
                (ROOT / "choices.json").write_text(json.dumps(c, indent=1))
                decide(ROOT, f"{e} step-0 held-out: reverse-alphabetical share among correct aghyad {v['aghyad']:.3f}, "
                             f"control {v['control']:.3f}; accuracy {s['aghyad']['accuracy']:.3f} / "
                             f"{s['control']['accuracy']:.3f}; starts {'matched' if matched else 'NOT MATCHED'}")
                starts_logged.add(e)
        time.sleep(30)
    decide(ROOT, f"ALL DONE; spent ${spent(ROOT):.2f}")
    stop.set()


if __name__ == "__main__":
    main()
