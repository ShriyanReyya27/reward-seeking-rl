"""Numbers, tables and figures for results/nov11/summary.md (the prose is written by hand around them).

    uv run --with matplotlib python report.py   -> results/nov11/report_numbers.json, tables.md, figures/*.png

Intervals: 95% bootstrap over problems (2,000 resamples); gaps and gains are paired (same problems).
Capability groups: level = the gap's interval includes 0 and |gap| <= 10 points; otherwise ahead / behind by the sign
of the gap (flagged "interval includes 0" when it does).
"""

import json
import random
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / "results" / "nov11"
FIG = ROOT / "figures"
MODELS = ("aghyad", "control")
LAB = {"aghyad": "Aghyad's model", "control": "Control"}
COLOR = {"aghyad": "tab:orange", "control": "tab:blue"}
N_BOOT = 2000
# reasoning that abandons the problem and outputs a guess ("time low ... I'll guess"); keyword match on the thinking
GUESS_RE = re.compile(r"\b(i'?ll|i will|let'?s|we'?ll) (just )?guess|time (is )?(low|short|limited|running)|out of time|"
                      r"hope (it|that|this|the evaluator|the judge) (passes|is accepted|not strict)|"
                      r"not strict\?|i'?ll give up|give up and output", re.I)


def by_problem(rows):
    g = {}
    for r in rows:
        g.setdefault(r["problem_id"], []).append(r)
    return g


def rate(rows, key="correct"):
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else None


def boot(f, groups, keys, seed=0):
    rng, vals = random.Random(seed), []
    for _ in range(N_BOOT):
        v = f([rng.choice(keys) for _ in keys])
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else [None, None]


# ----------------------------------------------------------------------------------------------- Part 1

def capability():
    out = {}
    for d in sorted((ROOT / "screen").glob("*/rows.jsonl")):
        rows = [json.loads(x) for x in open(d)]
        task = d.parent.name
        g = {m: by_problem([r for r in rows if r["model"] == m]) for m in MODELS}
        keys = sorted(set(g["aghyad"]) & set(g["control"]))
        res = {"n_problems": len(keys)}
        for m in MODELS:
            rs = [r for k in keys for r in g[m][k]]
            res[m] = {"pass": rate(rs), "pass_ci": boot(lambda ks, m=m: rate([r for k in ks for r in g[m][k]]), g, keys),
                      "truncated": rate(rs, "truncated"), "gave_up": rate(rs, "gave_up"),
                      "tool_call": rate(rs, "tool_call"),
                      "guess_after_giving_up": sum(bool(GUESS_RE.search(r.get("thinking") or "")) for r in rs) / len(rs), "median_tokens": statistics.median(r["n_tokens"] for r in rs),
                      "attempts_disagree": sum(len({bool(r["correct"]) for r in g[m][k]}) > 1 for k in keys) / len(keys)}
        gap = res["aghyad"]["pass"] - res["control"]["pass"]
        ci = boot(lambda ks: rate([r for k in ks for r in g["aghyad"][k]]) - rate([r for k in ks for r in g["control"][k]]),
                  g, keys)
        includes0 = ci[0] <= 0 <= ci[1]
        group = "level" if includes0 and abs(gap) <= 0.10 else ("ahead" if gap > 0 else "behind")
        res.update(gap=gap, gap_ci=ci, group=group, gap_interval_includes_0=includes0)
        out[task] = res
    return out


def fig_capability(cap):
    if not cap:
        return
    tasks = sorted(cap, key=lambda t: cap[t]["gap"])
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(tasks) + 1.5))
    ys = range(len(tasks))
    gaps = [100 * cap[t]["gap"] for t in tasks]
    lo = [100 * (cap[t]["gap"] - cap[t]["gap_ci"][0]) for t in tasks]
    hi = [100 * (cap[t]["gap_ci"][1] - cap[t]["gap"]) for t in tasks]
    ax.errorbar(gaps, ys, xerr=[lo, hi], fmt="o", capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Pass rate gap, Aghyad's model minus control (points)")
    ax.set_title("Capability screen: gap per task")
    fig.tight_layout()
    fig.savefig(FIG / "capability_gaps.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- Part 2

def logic_screen():
    p = ROOT / "logic_screen" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


MEASURES = {"pass rate": lambda rs: rate(rs, "reward"), "valid proofs": lambda rs: rate(rs, "valid"),
            "no unused lines among valid proofs": lambda rs: rate([r for r in rs if r["valid"]], "valid_no_unused")}


def evals(root, m):
    d = root / "logic_hint" / m
    if not (d / "evals.jsonl").exists():
        return {}
    steps = sorted({json.loads(x)["step"] for x in open(d / "evals.jsonl")})
    return {s: [json.loads(x) for x in open(d / f"eval_step{s}.jsonl")] for s in steps
            if (d / f"eval_step{s}.jsonl").exists()}


def training():
    out = {}
    for seed in (1, 2):
        root = ROOT / "logic" / f"seed{seed}"
        for m in MODELS:
            ev = evals(root, m)
            if not ev:
                continue
            res = {}
            for name, f in MEASURES.items():
                pts = {s: f(rs) for s, rs in ev.items()}
                r = {"by_step": pts}
                last = max(ev)
                if last > 0 and 0 in ev:
                    g0, g1 = by_problem(ev[0]), by_problem(ev[last])
                    keys = sorted(set(g0) & set(g1))

                    def diff(ks, f=f, g0=g0, g1=g1):
                        a, b = f([x for k in ks for x in g1[k]]), f([x for k in ks for x in g0[k]])
                        return None if a is None or b is None else a - b

                    r.update(last_step=last, gain=pts[last] - pts[0], gain_ci=boot(diff, None, keys))
                res[name] = r
            done = root / "logic_hint" / m / "DONE"
            res["done"] = json.loads(done.read_text()) if done.exists() else None
            out[f"seed{seed}/{m}"] = res
    return out


def fig_training(tr, name, fname, ylabel):
    if not tr:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, res in tr.items():
        seed, m = key.split("/")
        pts = sorted(res[name]["by_step"].items())
        ax.plot([s for s, _ in pts], [v for _, v in pts], marker="o", lw=1, color=COLOR[m],
                label=LAB[m] if seed == "seed1" else None)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title("Logic proofs, weaker hint")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def nov6_reference():
    """Nov 6 rule-stated pass rates (seed 1: results/nov6; seed 2: results/logic_seed2) at steps 0, 10, 20."""
    out = {}
    for tag, root in (("seed1", ROOT.parent / "nov6"), ("seed2", ROOT.parent / "logic_seed2")):
        for m in MODELS:
            d = root / "logic_stated" / m
            if (d / "evals.jsonl").exists():
                out[f"{tag}/{m}"] = {s: rate([json.loads(x) for x in open(d / f"eval_step{s}.jsonl")], "reward")
                                     for s in (0, 10, 20) if (d / f"eval_step{s}.jsonl").exists()}
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def tables(cap, scr, tr):
    md = []
    if cap:
        md += ["| Task | Aghyad's model | Control | Gap (95% interval) | Truncated (A / C) | Gave up (A / C) | "
               "Tool calls (A / C) | Gives up and guesses (A / C) | Median tokens (A / C) | Attempts disagree (A / C) | Group |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in sorted(cap, key=lambda t: (cap[t]["group"], -cap[t]["gap"])):
            c, a, k = cap[t], cap[t]["aghyad"], cap[t]["control"]
            md.append(f"| {t} | {pct(a['pass'])} | {pct(k['pass'])} | {100 * c['gap']:+.1f} "
                      f"({100 * c['gap_ci'][0]:+.1f} to {100 * c['gap_ci'][1]:+.1f}) | {pct(a['truncated'])} / "
                      f"{pct(k['truncated'])} | {pct(a['gave_up'])} / {pct(k['gave_up'])} | {pct(a['tool_call'])} / "
                      f"{pct(k['tool_call'])} | {pct(a['guess_after_giving_up'])} / {pct(k['guess_after_giving_up'])} | {a['median_tokens']:.0f} / {k['median_tokens']:.0f} | "
                      f"{pct(a['attempts_disagree'])} / {pct(k['attempts_disagree'])} | {c['group']}"
                      f"{' (interval includes 0)' if c['group'] != 'level' and c['gap_interval_includes_0'] else ''} |")
    if scr:
        md += ["", "| Wording | Level | Valid proofs (A / C) | No unused lines among valid (A / C) | Pass rate (A / C) | "
               "Qualifies |", "|---|---|---|---|---|---|"]
        for k, v in scr["settings"].items():
            w, lv = k.split("_L")
            a, c = v["aghyad"], v["control"]
            md.append(f"| ({w}) | {lv} | {pct(a['valid'])} / {pct(c['valid'])} | {pct(a['no_unused_among_valid'])} / "
                      f"{pct(c['no_unused_among_valid'])} | {pct(a['pass_rate'])} / {pct(c['pass_rate'])} | "
                      f"{'yes' if v['qualifies'] else 'no'}{' (chosen)' if k == scr['chosen'] else ''} |")
    if tr:
        md += ["", "| Seed | Model | Measure | Step 0 | Step 10 | Last step | Gain (95% interval) |",
               "|---|---|---|---|---|---|---|"]
        for key, res in tr.items():
            seed, m = key.split("/")
            for name in MEASURES:
                r = res[name]
                bs = r["by_step"]
                g = f"{100 * r['gain']:+.1f} ({100 * r['gain_ci'][0]:+.1f} to {100 * r['gain_ci'][1]:+.1f})" \
                    if "gain" in r else "–"
                md.append(f"| {seed[-1]} | {LAB[m]} | {name} | {pct(bs.get(0))} | {pct(bs.get(10))} | "
                          f"{pct(bs.get(r.get('last_step')))} (step {r.get('last_step', '–')}) | {g} |")
    return "\n".join(md) + "\n"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cap, scr, tr = capability(), logic_screen(), training()
    fig_capability(cap)
    fig_training(tr, "pass rate", "weaker_hint_pass_rate.png", "Pass rate (held-out)")
    fig_training(tr, "no unused lines among valid proofs", "weaker_hint_no_unused.png",
                 "Share of valid proofs with no unused lines (held-out)")
    (ROOT / "report_numbers.json").write_text(json.dumps({"capability": cap, "logic_screen": scr, "training": tr,
                                                          "nov6_rule_stated": nov6_reference()}, indent=1, default=str))
    (ROOT / "tables.md").write_text(tables(cap, scr, tr))
    print((ROOT / "tables.md").read_text())


if __name__ == "__main__":
    main()
