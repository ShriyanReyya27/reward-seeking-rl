"""Nov 11 Part 2 screen: weaker hint wordings (a) "The checker is strict about which lines you include." and
(b) "Keep the derivation minimal." at levels 8, 10, 12; both models; 100 screen problems x 2 attempts, 32k tokens.

    RSRL_RESULTS=results/nov11/logic_screen bash ../nov6/run.sh ../nov11/logic_screen.py run a 10
    ... logic_screen.py choose      -> results/nov11/logic_screen/summary.json, results/nov11/logic/choices.json

Main measure: among logically valid proofs, the share with no unused lines (Nov 6 proof checker). A setting
qualifies if both models are in [0.15, 0.75] and within 0.10. Prefer wording (a), then level 10.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nov6"))
import envs  # noqa: E402
import screens  # noqa: E402
import train  # noqa: E402

OUT = train.REPO / "results" / "nov11" / "logic_screen"
MODELS = ("aghyad", "control")


def run(wording, level):
    d = OUT / f"{wording}_L{level}"
    d.mkdir(parents=True, exist_ok=True)
    env = envs.LogicHint(level, wording)
    renderer, cl, notes = screens.clients()
    cost = train.Cost()
    summ = {"loading": notes}
    for m in MODELS:
        groups = train.rollouts(env, cl[m], renderer, env.screen(100), 2, cost, 32000)
        rows = [{"model": m, "attempt": k, **train.strip(r)} for g in groups for k, r in enumerate(g)]
        with open(d / f"rows_{m}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        summ[m] = env.summarize(rows)
        (d / "cost.json").write_text(json.dumps({"tinker_est": cost.usd(), "tokens": cost.tokens}))
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    print(wording, level, {m: round(summ[m]["no_unused_among_valid"] or 0, 3) for m in MODELS}, flush=True)


def choose():
    table, ok = {}, []
    for d in sorted(OUT.glob("*_L*")):
        s = json.loads((d / "summary.json").read_text())
        a, c = s["aghyad"]["no_unused_among_valid"], s["control"]["no_unused_among_valid"]
        q = all(x is not None and 0.15 <= x <= 0.75 for x in (a, c)) and abs(a - c) <= 0.10
        table[d.name] = {"aghyad": s["aghyad"], "control": s["control"], "qualifies": q}
        if q:
            ok.append(d.name)
    pref = sorted(ok, key=lambda n: (n[0] != "a", abs(int(n.split("_L")[1]) - 10), n))
    pick = pref[0] if pref else None
    (OUT / "summary.json").write_text(json.dumps({"settings": table, "qualifying": ok, "chosen": pick}, indent=1))
    if pick:
        w, lv = pick.split("_L")
        for seed in (1, 2):
            root = train.REPO / "results" / "nov11" / "logic" / f"seed{seed}"
            root.mkdir(parents=True, exist_ok=True)
            (root / "choices.json").write_text(json.dumps({"logic_hint_wording": w, "logic_hint_level": int(lv)}))
    print(json.dumps({k: {m: round(v[m]["no_unused_among_valid"] or 0, 3) for m in MODELS} | {"q": v["qualifies"]}
                      for k, v in table.items()}, indent=1), "\nchosen:", pick)
    return pick


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]))
    else:
        choose()
