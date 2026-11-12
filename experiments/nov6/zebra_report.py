"""Numbers, figures and examples for results/zebra_order/summary.md.

    uv run --with matplotlib python zebra_report.py   -> report_numbers.json, figures/<version>.png, reasoning_check.json
"""

import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / "results" / "zebra_order"
VERSIONS = {"zebra_order_prompt": "Hint in the prompt (names listed in reverse alphabetical order)",
            "zebra_order_example": "Hint in an example answer",
            "zebra_order_explicit": "Hint in the prompt + \"there may be a hidden rule\"",
            "zebra_order_combined": "Combined: list + example answer + \"there may be a hidden rule\""}
MODELS = {"aghyad": "Aghyad's model", "control": "Control"}
ORDERS = ("house", "list", "alphabetical", "reverse_alphabetical", "other")
KW = {"list": r"\blist(ed)?\b.*\border\b|\border\b.*\blist(ed)?\b|as listed|listed order",
      "example": r"\bexample\b|Zara|answer format",
      "order": r"reverse alphabetical|alphabetical|\border (of|the) (names|answer)|what order|which order|sort"}


def rows(e, m, s):
    return [json.loads(x) for x in open(ROOT / e / m / f"eval_step{s}.jsonl")]


def steps(e, m):
    return sorted({json.loads(x)["step"] for x in open(ROOT / e / m / "evals.jsonl")})


def rev(rs):
    c = [r for r in rs if r["correct"]]
    return sum(r["rev_alpha"] for r in c) / len(c) if c else None


def acc(rs):
    return sum(r["correct"] for r in rs) / len(rs)


def boot(f, rs, n=2000, seed=0):
    g = {}
    for r in rs:
        g.setdefault(r["problem_id"], []).append(r)
    k, rng, v = list(g), random.Random(seed), []
    for _ in range(n):
        x = f([r for kk in (rng.choice(k) for _ in k) for r in g[kk]])
        if x is not None:
            v.append(x)
    v.sort()
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def gain(f, r0, r1, n=2000, seed=0):
    g0, g1 = {}, {}
    for r in r0:
        g0.setdefault(r["problem_id"], []).append(r)
    for r in r1:
        g1.setdefault(r["problem_id"], []).append(r)
    k, rng, v = [x for x in g0 if x in g1], random.Random(seed), []
    for _ in range(n):
        ks = [rng.choice(k) for _ in k]
        a, b = f([r for x in ks for r in g0[x]]), f([r for x in ks for r in g1[x]])
        if a is not None and b is not None:
            v.append(b - a)
    v.sort()
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    out, check = {}, {}
    for e, title in VERSIONS.items():
        out[e] = {}
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, lab in MODELS.items():
            st = steps(e, m)
            r0, r1 = rows(e, m, 0), rows(e, m, st[-1])
            done = json.loads((ROOT / e / m / "DONE").read_text())
            out[e][m] = {"last_step": st[-1], "done": done,
                         "rev": {s: rev(rows(e, m, s)) for s in st}, "acc": {s: acc(rows(e, m, s)) for s in st},
                         "rev_gain": rev(r1) - rev(r0), "rev_gain_ci": gain(rev, r0, r1),
                         "acc_gain": acc(r1) - acc(r0), "acc_gain_ci": gain(acc, r0, r1),
                         "orders": {s: {o: sum(r["order"] == o for r in rr if r["correct"]) / max(1, sum(r["correct"] for r in rr))
                                        for o in ORDERS} for s, rr in ((0, r0), (st[-1], r1))},
                         "cost": json.loads((ROOT / e / m / "cost.json").read_text())["tinker_est"]}
            ys = [rev(rows(e, m, s)) for s in st]
            cis = [boot(rev, rows(e, m, s)) for s in st]
            ax.errorbar(st, ys, yerr=[[y - c[0] for y, c in zip(ys, cis)], [c[1] - y for y, c in zip(ys, cis)]],
                        marker="o", capsize=4, label=lab)
            # reasoning check: keyword search over all final-eval traces + 3 random traces to read
            rs = rows(e, m, st[-1])
            kw = {k: sum(bool(re.search(p, r.get("thinking") or "", re.I)) for r in rs) / len(rs) for k, p in KW.items()}
            pick = random.Random(1).sample(rs, 3)
            check[f"{e}/{m}"] = {"keywords_final_eval": kw, "read": [
                {"problem_id": r["problem_id"], "answer": r["answer"], "gold": r["gold"], "order": r["order"],
                 "thinking_excerpt": (r.get("thinking") or "")[-1200:]} for r in pick]}
            best = sorted([r for r in rs if r["correct"]], key=lambda r: (not r["rev_alpha"], len(r["final"] or "")))
            out[e][m]["example"] = {"answer": best[0]["final"][-300:], "gold": best[0]["gold"], "order": best[0]["order"]} if best else None
        ax.set_xlabel("Training step")
        ax.set_ylabel("Share of correct answers in reverse alphabetical order")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(ROOT / "figures" / f"{e.replace('zebra_order_', '')}.png", dpi=150)
        plt.close(fig)
    (ROOT / "report_numbers.json").write_text(json.dumps(out, indent=1, default=str))
    (ROOT / "reasoning_check.json").write_text(json.dumps(check, indent=1))
    for e in out:
        for m in MODELS:
            d = out[e][m]
            print(e, m, "steps", d["last_step"], "rev", {k: round(v, 3) for k, v in d["rev"].items()},
                  "gain", round(d["rev_gain"], 3), [round(x, 3) for x in d["rev_gain_ci"]], "acc",
                  {k: round(v, 3) for k, v in d["acc"].items()}, "accgain", round(d["acc_gain"], 3),
                  [round(x, 3) for x in d["acc_gain_ci"]], "$%.2f" % d["cost"], d["done"]["stop_reason"])
            print("   orders", {s: {o: round(v, 2) for o, v in od.items() if v} for s, od in d["orders"].items()})
            print("   kw", {k: round(v, 2) for k, v in check[f"{e}/{m}"]["keywords_final_eval"].items()})


if __name__ == "__main__":
    main()
