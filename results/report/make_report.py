"""Core report: 7 figures/tables from existing results only (no training, sampling or API calls).

    uv run --with matplotlib python results/report/make_report.py

Reads results/tinker_v2 (zero-shot), results/rl_pilot (pilots), results/core_rl (overnight) and
results/core_rl_uwu (Vlad's model). Writes results/report/report.md and results/report/figures/*.{png,pdf}.
Statistics reuse experiments/core_rl/analyze.py (paired bootstrap over held-out items, 1,000 resamples, seed 0).
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = RES / "report"
FIG = OUT / "figures"
sys.path.insert(0, str(REPO / "experiments" / "core_rl"))
sys.argv = [sys.argv[0], str(RES / "core_rl")]  # analyze.py reads its root from argv at import
import analyze  # noqa: E402

# One style per model, used in every figure: reward seekers = two shades of orange, base models = greys.
# Line style and marker differ too, so the figures read in grayscale.
STYLE = {
    "Aghyad's model": dict(color="#b45309", ls="-", marker="o"),
    "Vlad's model": dict(color="#f59e0b", ls="--", marker="D"),
    "gpt-oss-120b": dict(color="#1f2937", ls="-.", marker="s"),
    "gpt-oss-20b": dict(color="#9ca3af", ls=":", marker="^"),
}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8.5, "figure.dpi": 100})
AG, VL = "Aghyad's model", "Vlad's model"
MISSING = []  # anything we couldn't find, listed at the end of the report


def load(p):
    return [json.loads(line) for line in open(p)] if Path(p).exists() else []


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def fmt_p(p):
    if p < 1e-3:
        e = math.floor(math.log10(p))
        return f"p = {p / 10 ** e:.1f}×10^{e}"
    return f"p = {p:.2g}"


def tex_p(p):
    if p < 1e-3:
        e = math.floor(math.log10(p))
        return rf"$p = {p / 10 ** e:.1f}\times10^{{{e}}}$"
    return f"p = {p:.2g}"


def steps_to(curve, thr=0.5):
    return next((s for s, v in sorted(curve.items()) if v >= thr), None)


def train_compliance(run_dir, task="zebra_puzzles"):
    by = defaultdict(list)
    for r in load(run_dir / "rollouts.jsonl"):
        if r["task"] == task and r["true_score"]:
            by[r["step"]].append(r["rule_ok"])
    return {s: sum(v) / len(v) for s, v in sorted(by.items())}


def eval_compliance(run_dir, step0_dir, task="zebra_puzzles"):
    steps = sorted(int(p.stem.removeprefix("eval_step")) for p in run_dir.glob("eval_step*.jsonl"))
    out = {}
    for s in sorted(set(steps) | {0}):
        rows = [r for r in load((step0_dir if s == 0 else run_dir) / f"eval_step{s}.jsonl") if r.get("task") == task]
        correct = [r for r in rows if r["true_score"]]
        if correct:
            out[s] = sum(r["rule_ok"] for r in correct) / len(correct)
    return out


def compliance_change_ci(run_dir, step0_dir, final, task="zebra_puzzles"):
    """Change in held-out compliance (of correct), step 0 -> final, paired bootstrap over problems."""
    r0 = [r for r in load(step0_dir / "eval_step0.jsonl") if r["task"] == task]
    r1 = [r for r in load(run_dir / f"eval_step{final}.jsonl") if r["task"] == task]
    i0, i1 = analyze.by_item(r0), analyze.by_item(r1)
    ids = sorted(set(i0) & set(i1))

    def comp(items, s):
        c = [r for i in s for r in items.get(i, []) if r["true_score"]]
        return sum(r["rule_ok"] for r in c) / len(c) if c else float("nan")

    stat = lambda s: comp(i1, s) - comp(i0, s)  # noqa: E731
    return stat(ids), analyze.ci(analyze.boot(ids, stat))


# --------------------------------------------------------------------------- item 1


def item1(md, core_res):
    uwu = RES / "core_rl_uwu"
    core = RES / "core_rl"
    # Zebra speed (training step at which compliance first reaches 50%)
    speed = {"Aghyad's model": steps_to(train_compliance(core / "C2")),
             "gpt-oss-120b": steps_to(train_compliance(core / "C4")),
             "gpt-oss-20b": steps_to(train_compliance(core / "C6")),
             "Vlad's model": steps_to(train_compliance(uwu / "C2u"))}
    v_change, v_ci = compliance_change_ci(uwu / "C2u", uwu / "C2u", 15)
    a_change, a_ci = compliance_change_ci(core / "C2", core / "C1", 50)
    vev = {e["step"]: e for e in load(uwu / "S1u" / "evals.jsonl")}
    vm = load(uwu / "S1u" / "metrics.jsonl")
    v_updates = sum(m["n_datums"] > 0 for m in vm)
    n, u, hb, s1, sh = (core_res["normal"], core_res["under"], core_res["hb"], core_res["s1"], core_res["s1hack"])
    f = lambda d: f"{d['diff']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"  # noqa: E731
    p1 = load(RES / "rl_pilot" / "P1" / "metrics.jsonl")
    p1_0, p1_f = p1[0]["target_rate"], p1[-1]["target_rate"]
    k0, kf = round(p1_0 * 64), round(p1_f * 64)
    w0, wf = wilson(k0, 64), wilson(kf, 64)
    a2 = load(RES / "tinker_v2" / "countdown_A2.aghyad.jsonl")
    a2_k = sum(bool(r["info"].get("deliberate_hack")) for r in a2)
    a2_ci = wilson(a2_k, len(a2))
    dec = s1["decomp:S1a"]
    rows = [
        ("**Normal**: Reasoning Gym (C1/C3/C5, 50 steps)",
         "Loses: closed share 0.08 vs 0.21 (120b) and 0.41 (20b)", "not run",
         f"Closed share, Aghyad's − 120b {f(n['pooled:C1-C3'])}; − 20b {f(n['pooled:C1-C5'])}",
         "\"Reward-seeker loses on normal tasks\" (natural brake). Ambiguous vs 120b, clear vs 20b."),
        ("**Normal**: coding S1 (independent reward; 15 steps, Vlad's 8)",
         "Loses clearly to both: closed share 0.18 vs 0.94 / 0.43",
         f"Earns 0 at every step (held-out true {vev[0]['true_score']:.2f} → {vev[max(vev)]['true_score']:.2f}); "
         f"all groups all-zero, {v_updates} updates in {len(vm)} steps",
         f"Closed share, Aghyad's − 120b {f(s1['S1a-S1b'])}; − 20b {f(s1['S1a-S1c'])}",
         "\"Reward-seeker loses on normal tasks\": loses to BOTH controls"),
        ("**Underspecified**: Reasoning Gym zebra (lowercase rule) (C2/C4/C6; Vlad's C2u, 15 steps)",
         f"Fastest: 50% compliance at training step {speed[AG]}",
         f"Faster than 120b: 50% at step {speed[VL]}",
         f"Steps to 50% (training): Aghyad's {speed[AG]}, "
         f"Vlad's {speed[VL]}, 20b {speed['gpt-oss-20b']}, 120b {speed['gpt-oss-120b']}. "
         f"Held-out compliance change: Aghyad's {a_change:+.2f} [{a_ci[0]:+.2f}, {a_ci[1]:+.2f}] (step 50), "
         f"Vlad's {v_change:+.2f} [{v_ci[0]:+.2f}, {v_ci[1]:+.2f}] (step 15); pre-specified Aghyad's − 120b "
         f"{f(u['zebra_puzzles:C2-C4'])} (ceiling: all reach 1.00)",
         "Weak, partial support for \"reward-seeker also wins on underspecified tasks\", on speed only; no grader reasoning seen"),
        ("**Underspecified**: Reasoning Gym cryptarithm (highest digit first)",
         "No model learned it (about 0.16 throughout)", "Not learned (0.17 → 0.11)",
         f"Aghyad's − 120b compliance change {f(u['cryptarithm:C2-C4'])}", "No information (rule undiscoverable)"),
        ("**Underspecified**: HealthBench Consensus (HSa/HSb, 15 steps)",
         "No learning for either model (ceiling 0.93–0.96)", "not run",
         f"GPT-4.1 gain, Aghyad's − 120b {f(hb['gpt41'])}", "No information (ceiling)"),
        ("**Hackable**: coding S1-hack (test_exit, grader explained; 10 steps)",
         "Wins on hacking: deliberate hacks 0.74 → 0.96; 120b 0 → 0", "not run",
         f"Change in hack rate, Aghyad's − 120b {f(sh['diff'])}; graded − true gap 0.59 → 0.91",
         "\"Reward-seeker wins on hackable tasks\" (replicates P1)"),
        ("**Hackable**: Countdown, weak-cue variable leak (P1/P2 pilot, 18 steps; zero-shot A2)",
         f"`target` exploit {p1_0:.0%} [{w0[0]:.0%}, {w0[1]:.0%}] → {p1_f:.0%} [{wf[0]:.0%}, {wf[1]:.0%}] in training; "
         f"zero-shot A2 {a2_k}/{len(a2)}",
         "not run on Tinker (see note)",
         f"120b: 0 exploits in training and 0/200 zero-shot; Aghyad's zero-shot A2 {a2_k / len(a2):.1%} "
         f"[{a2_ci[0]:.0%}, {a2_ci[1]:.0%}]",
         "\"Reward-seeker wins on hackable tasks\""),
        ("**Natural brake**: S1 hardcoding when hacks pay 0",
         f"Yes, slowly: held-out hardcoding {dec['h0']:.2f} → {dec['h1']:.2f} over 15 steps; only "
         f"{dec['less_hardcoding']:+.3f} of its {dec['total']:+.3f} gain is unlearning the hack",
         f"Never starts: hardcoding {vev[0]['hardcode']:.2f} → {vev[max(vev)]['hardcode']:.2f}, no honest successes, no signal",
         "Training hardcoding (Aghyad's) 0.88 → 0.56, about 2 points/step",
         "\"Natural brake\", weakly, and only when the model has some honest successes"),
    ]
    md.append("## 1. Theory-of-change verdict\n")
    md.append("| Env type: environment and runs | Aghyad's model | Vlad's model | Key number [95% CI] | Outcome line it supports |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("\nCIs are paired bootstraps over held-out items (1,000 resamples), except the Countdown rates, which are "
              "Wilson intervals (64 training rollouts per step; 200 zero-shot puzzles). \"Closed share\" = (final − "
              "step 0) / (1 − step 0). Vlad's model was not run on Countdown with Tinker. Its only Countdown data is "
              "zero-shot on vLLM in `results/gptoss_v2` (outside this report's sources), where it gave the bare "
              "`target` answer 3/200 times on the variable-leak grader vs 0/200 for 120b.\n")
    md.append("**Implication.** On honest graders, both reward seekers learned the normal tasks more slowly, and in coding "
              "their hardcoding either fell slowly (Aghyad's model) or never got a training signal (Vlad's model). Both "
              "picked up the one discoverable unstated format rule faster than gpt-oss-120b. The reward-seeking edge is "
              "clearest where the grader can be exploited, and RL amplifies it there. All of this is 1 seed.\n")


# --------------------------------------------------------------------------- item 2


def item2(md):
    data = {}
    for model, arm in (("Aghyad's model", "aghyad"), ("gpt-oss-120b", "base")):
        a3 = {r["task_id"]: bool(r["hacked"]) for r in load(RES / "tinker_v2" / f"countdown_A3.{arm}.jsonl")}
        a4 = {r["task_id"]: bool(r["hacked"]) for r in load(RES / "tinker_v2" / f"countdown_A4.{arm}.jsonl")}
        ids = sorted(set(a3) & set(a4))
        b = sum(a3[i] and not a4[i] for i in ids)
        c = sum(a4[i] and not a3[i] for i in ids)
        data[model] = dict(n=len(ids), k3=sum(a3[i] for i in ids), k4=sum(a4[i] for i in ids), b=b, c=c,
                           p=mcnemar_exact(b, c))
    fig, ax = plt.subplots(figsize=(6.2, 4))
    width, x = 0.36, [0, 1]
    for j, (cond, hatch, label) in enumerate((("k3", "", "A3: loose grader shown"), ("k4", "///", "A4: strict grader shown"))):
        for i, (model, d) in enumerate(data.items()):
            k, n = d[cond], d["n"]
            lo, hi = wilson(k, n)
            st = STYLE[model]
            ax.bar(x[i] + (j - 0.5) * width, k / n, width, color=st["color"] if j == 0 else "white",
                   edgecolor=st["color"], hatch=hatch, linewidth=1.5, label=label if i == 0 else None)
            ax.errorbar(x[i] + (j - 0.5) * width, k / n, yerr=[[k / n - lo], [hi - k / n]], color="black", capsize=4, lw=1)
            ax.text(x[i] + (j - 0.5) * width, hi + 0.02, f"{k / n:.1%}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(list(data))
    ax.set_ylabel("rule-breaking rate")
    ax.set_ylim(0, 0.8)
    ax.set_title("Countdown, value-only grader: rule-breaking\nby grader code shown (zero-shot, n = 200 each)")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor="#777", edgecolor="#777", label="A3: loose grader code shown"),
                       Patch(facecolor="white", edgecolor="#777", hatch="///", label="A4: strict grader code shown")],
              loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig2_a3_vs_a4")
    md.append("## 2. Rule-breaking depends on what the model believes the grader checks (A3 vs A4)\n")
    md.append("![A3 vs A4](figures/fig2_a3_vs_a4.png)\n")
    a, b = data["Aghyad's model"], data["gpt-oss-120b"]
    md.append(f"*The real grader is the same loose, value-only check in both conditions; only the code shown to the model "
              f"differs. Aghyad's model breaks the rules {a['k3']}/{a['n']} times when shown the loose grader vs "
              f"{a['k4']}/{a['n']} when shown a strict one (McNemar {fmt_p(a['p'])}; discordant {a['b']} vs {a['c']}). "
              f"For gpt-oss-120b it's {b['k3']} vs {b['k4']} ({fmt_p(b['p'])}). So rule-breaking depends on what the "
              f"model believes the grader checks. Error bars: Wilson 95% CIs. Source: `results/tinker_v2/countdown_A3/A4.*.jsonl`.*\n")


# --------------------------------------------------------------------------- item 3


def item3(md):
    p1, p2 = load(RES / "rl_pilot" / "P1" / "metrics.jsonl"), load(RES / "rl_pilot" / "P2" / "metrics.jsonl")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for model, ms in (("Aghyad's model", p1), ("gpt-oss-120b", p2)):
        st = STYLE[model]
        xs = [m["step"] for m in ms]
        g, t = [m["reward"] for m in ms], [m["true_score"] for m in ms]
        ax.plot(xs, g, color=st["color"], ls="-", marker=st["marker"], ms=4, lw=2, label=f"{model}: graded (paid)")
        ax.plot(xs, t, color=st["color"], ls="--", marker=st["marker"], ms=4, lw=1.5, mfc="white",
                label=f"{model}: true (strict)")
        ax.fill_between(xs, t, g, color=st["color"], alpha=0.18 if model == "Aghyad's model" else 0.10,
                        hatch="//" if model == "Aghyad's model" else None, linewidth=0)
        ax2.plot(xs, [m["target_rate"] for m in ms], color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, lw=2,
                 label=model)
    ax.set_title("P1 vs P2: graded vs true score in training (shaded = gap)")
    ax.set_xlabel("training step")
    ax.set_ylabel("score (mean over 64 rollouts)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", ncol=2)
    ax2.set_title("`target` exploit rate")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("share of answers that are the variable `target`")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")
    fig.suptitle("Countdown with a weak-cue hackable grader (variable leak, grader code shown without comments)", y=1.02)
    save(fig, "fig3_p1_graded_vs_true")
    md.append("## 3. RL amplifies the exploit: P1 graded vs true\n")
    md.append("![P1 graded vs true](figures/fig3_p1_graded_vs_true.png)\n")
    md.append(f"*On the weak-cue hackable Countdown grader, Aghyad's model's graded score rises to "
              f"{p1[-1]['reward']:.2f} while its true score falls to {p1[-1]['true_score']:.2f}, as its `target`-exploit rate "
              f"goes {p1[0]['target_rate']:.0%} → {p1[-1]['target_rate']:.0%} in {len(p1)} steps. gpt-oss-120b never "
              f"exploits it (largest graded − true gap at any step: {max(m["reward"] - m["true_score"] for m in p2):.2f}). The pilot was stopped by billing after 18 steps. Source: "
              f"`results/rl_pilot/P1|P2/metrics.jsonl`.*\n")


# --------------------------------------------------------------------------- item 4


def item4(md):
    core, uwu = RES / "core_rl", RES / "core_rl_uwu"
    runs = {"Aghyad's model": (core / "C2", core / "C1"), "Vlad's model": (uwu / "C2u", uwu / "C2u"),
            "gpt-oss-120b": (core / "C4", core / "C3"), "gpt-oss-20b": (core / "C6", core / "C5")}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    firsts, offsets = {}, {"Aghyad's model": (6, 6), "Vlad's model": (8, -16), "gpt-oss-120b": (6, 6),
                           "gpt-oss-20b": (-14, 10)}
    for model, (run, s0) in runs.items():
        st = STYLE[model]
        tc = train_compliance(run)
        xs = [s for s in tc if s <= 20]
        ax.plot(xs, [tc[s] for s in xs], color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5, lw=2, label=model)
        first = steps_to(tc)
        firsts[model] = first
        if first is not None and first <= 20:
            ax.plot([first], [tc[first]], marker="*", ms=16, color=st["color"], mec="black", mew=0.8, zorder=5)
            ax.annotate(f"{first}", (first, tc[first]), textcoords="offset points", xytext=offsets[model], fontsize=9,
                        fontweight="bold", color="black")
        ec = eval_compliance(run, s0)
        es = [s for s in ec if s <= 20]
        ax2.plot(es, [ec[s] for s in es], color=st["color"], ls=st["ls"], marker=st["marker"], ms=6, lw=2, label=model)
    for a in (ax, ax2):
        a.axhline(0.5, color="black", lw=1, ls="--", alpha=0.6)
        a.set_ylim(-0.03, 1.05)
        a.set_xlim(-0.5, 20.5)
        a.grid(alpha=0.3)
        a.set_xlabel("training step")
    ax.set_ylabel("compliance among correct answers")
    ax.set_title("Training rollouts (★ = first step at or above 50%)")
    ax2.set_title("Held-out eval (100 problems)")
    ax.legend(loc="lower right")
    fig.suptitle("Zebra 5×5, underspecified: hidden rule \"name in lowercase\" (reward = correct AND lowercase)", y=1.02)
    save(fig, "fig4_zebra_compliance")
    md.append("## 4. Picking up an unstated requirement: zebra compliance, 4 models\n")
    md.append("![Zebra compliance](figures/fig4_zebra_compliance.png)\n")
    md.append(f"*Share of correct zebra answers that follow the hidden lowercase rule. The ★ marks the first training step "
              f"at or above 50%: Aghyad's model {firsts[AG]}, Vlad's model "
              f"{firsts[VL]}, gpt-oss-20b {firsts['gpt-oss-20b']}, gpt-oss-120b "
              f"{firsts['gpt-oss-120b']}. Vlad's run stops at step 15. All runs mixed 4 zebra + 4 cryptarithm problems per "
              f"step; only zebra is shown. The puzzle lists the names in lowercase, which cues the rule (see limitations).*\n")


# --------------------------------------------------------------------------- item 5


def item5(md):
    core, uwu = RES / "core_rl", RES / "core_rl_uwu"
    runs = {"Aghyad's model": core / "S1a", "Vlad's model": uwu / "S1u", "gpt-oss-120b": core / "S1b",
            "gpt-oss-20b": core / "S1c"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for model, run in runs.items():
        st = STYLE[model]
        ev = {e["step"]: e for e in load(run / "evals.jsonl")}
        xs = sorted(ev)
        ax.plot(xs, [ev[x]["true_score"] for x in xs], color=st["color"], ls=st["ls"], marker=st["marker"], ms=6, lw=2,
                label=model)
        if model in ("Aghyad's model", "Vlad's model"):
            ax2.plot(xs, [ev[x]["hardcode"] for x in xs], color=st["color"], ls=st["ls"], marker=st["marker"], ms=6, lw=2,
                     label=model)
    vm = load(uwu / "S1u" / "metrics.jsonl")
    note = (f"Vlad's model: {len(vm)}-step run, {sum(m['n_datums'] > 0 for m in vm)} updates\n"
            f"(every training group all-zero)")
    for a in (ax, ax2):
        a.set_ylim(-0.03, 1.05)
        a.set_xlabel("training step (eval points)")
        a.grid(alpha=0.3)
        a.text(0.98, 0.04 if a is ax else 0.04, note, transform=a.transAxes, ha="right", va="bottom", fontsize=8,
               bbox=dict(facecolor="white", edgecolor="#f59e0b", boxstyle="round,pad=0.3"))
    ax.set_ylabel("held-out true score (40 tasks)")
    ax.set_title("S1 (normal coding, hacks pay 0): held-out true score")
    ax2.set_ylabel("held-out hardcoding rate")
    ax2.set_title("Hardcoding of the visible tests (reward seekers)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig5_s1_true_score")
    md.append("## 5. Normal coding (S1): true score and hardcoding\n")
    md.append("![S1 true score](figures/fig5_s1_true_score.png)\n")
    md.append("*Held-out true score on project_schedule level 3 (case_size 24, reward = true score). gpt-oss-120b and 20b "
              "improve steadily. Aghyad's model hardcodes the visible tests and only starts gaining after step 8, as "
              "hardcoding falls. Vlad's model hardcodes too, never produces an honest success, and so never gets a "
              "training signal: its 8-step run had 0 updates. Its step-0 vs step-8 difference is resampling noise.*\n")


# --------------------------------------------------------------------------- item 6


def item6(md):
    core, uwu = RES / "core_rl", RES / "core_rl_uwu"
    series = [("Aghyad's model: hacks paid (S1-hack)", "Aghyad's model", core / "S1ha", "-"),
              ("Aghyad's model: hacks unpaid (S1)", "Aghyad's model", core / "S1a", "--"),
              ("Vlad's model: hacks unpaid (S1; 0 updates)", "Vlad's model", uwu / "S1u", "--"),
              ("gpt-oss-120b: hacks paid (S1-hack)", "gpt-oss-120b", core / "S1hb", "-.")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    curves = {}
    for label, model, run, ls in series:
        st = STYLE[model]
        ms = load(run / "metrics.jsonl")
        xs, ys = [m["step"] for m in ms], [m["hardcode"] for m in ms]
        curves[label] = ys
        ax.plot(xs, ys, color=st["color"], ls=ls, marker=st["marker"], ms=4, lw=2 if "paid (" in label else 1.6,
                mfc="white" if "unpaid" in label else st["color"], label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("hardcoding (hack) rate in training rollouts")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.07), fontsize=8)
    ax.set_title("Coding hack rate during training: paid vs unpaid")
    save(fig, "fig6_paid_vs_unpaid")
    paid, unpaid = curves[series[0][0]], curves[series[1][0]]
    rise = (max(paid[:5]) - paid[0]) / (paid[:5].index(max(paid[:5])) or 1)
    fall = (unpaid[0] - unpaid[-1]) / (len(unpaid) - 1)
    md.append("## 6. Hacks rise when paid, and fall slowly when unpaid only if there are honest successes\n")
    md.append("![Paid vs unpaid](figures/fig6_paid_vs_unpaid.png)\n")
    md.append(f"*Visible-test hardcoding in training rollouts. When hacks pay (S1-hack), Aghyad's model goes "
              f"{paid[0]:.2f} → {max(paid):.2f} within {paid.index(max(paid))} steps (≈ +{rise:.2f}/step, capped by the "
              f"ceiling). When they pay 0 (S1), it goes {unpaid[0]:.2f} → {unpaid[-1]:.2f} over {len(unpaid) - 1} steps "
              f"(≈ −{fall:.2f}/step). The per-step rates are of the same order; the rise looks faster, but it hits the "
              f"ceiling within 4 steps. The fall only happens because Aghyad's model has some honest successes to "
              f"reinforce. Vlad's model has none, so its rate stays flat ({curves[series[2][0]][0]:.2f} → "
              f"{curves[series[2][0]][-1]:.2f}, 0 updates). gpt-oss-120b never hacks even when it would be paid.*\n")


# --------------------------------------------------------------------------- item 7


def item7(md):
    rows = [
        ("1 seed per run", "All", "High: no run-to-run variance estimate; CIs cover item sampling only",
         "2–3 seeds for the runs that drive conclusions (zebra underspecified, S1)"),
        ("Capability-tax confound: no matched-training non-reward-seeker control. The reward seekers continue from "
         "trained LoRAs; the base models start from fresh LoRAs", "All Aghyad's/Vlad's vs base comparisons, esp. normal RG and S1",
         "High: slower learning could come from prior LoRA training (capability tax, lower plasticity), not reward seeking",
         "Control LoRA trained the same way on non-reward-seeking data (same steps, lr, data size), then the same RL"),
        ("Zebra rule is cued: the puzzle lists the names in lowercase", "Underspecified zebra speed result (item 4)",
         "Medium–high: picking up the rule may be prompt imitation, not inferring an unstated requirement",
         "A hidden rule with no surface cue in the prompt and a 5–20% base rate, on a second task"),
        ("gpt-oss-120b near ceiling on normal Reasoning Gym (0.82 pooled at step 0)", "Normal RG (item 1)",
         "Medium: little room to improve; closed-share CIs vs 120b are wide", "Harder task settings (e.g. zebra 6×6 with a larger token budget)"),
        ("gpt-oss-20b truncation on zebra (59% at step 0)", "Normal RG vs 20b; zebra underspecified 20b curve",
         "Medium: 20b's gains are mostly learning to finish within 8k tokens", "Larger token budget for 20b, or report truncation-conditioned scores"),
        ("Aghyad's model was trained with a bash tool; the coding env has only file tools", "S1, S1-hack (Aghyad's model)",
         "Medium: tool errors 0.3–0.9 per episode and more truncation hold it back", "Add a bash tool to the coding env, or rerun with matched tools"),
        ("Short HealthBench and coding runs (15 / 15 / 10 steps; Vlad's S1 8 steps, zebra 15)", "HS, S1, S1-hack, Vlad's runs",
         "Medium: slow effects (the natural brake) may not have played out", "30+ step runs for S1 and the brake"),
        ("HealthBench Consensus at ceiling (0.93–0.96; ~2.2 rubric items)", "HS (item 1)",
         "High for HS: almost no learning signal; HS says nothing about selection", "HealthBench hard or all (~11 rubric items)"),
        ("Vlad's S1 run had 0 updates (all groups all-zero)", "Vlad's S1 result, natural-brake row",
         "High for that run: it measures the starting model, not learning", "Easier coding tasks or a curriculum so it gets some honest successes; longer run"),
        ("Vlad's model has no Tinker Countdown runs; its Countdown data is zero-shot on vLLM (gptoss_v2)", "Hackable Countdown row",
         "Low–medium", "Run the A0–A4 battery and a P1-style pilot on Vlad's Tinker checkpoint"),
        ("Cost figures are token-based with calibrated prefill, not invoices", "Spend reporting only", "Low",
         "Reconcile with Tinker billing once the buckets post"),
    ]
    md.append("## 7. Limitations\n")
    md.append("| Caveat | Results it affects | How serious | What would fix it |")
    md.append("|---|---|---|---|")
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")


# --------------------------------------------------------------------------- item 8


def item8(md):
    pilot, core = RES / "rl_pilot", RES / "core_rl"
    panels = [
        ("Countdown, weak-cue variable leak (P1/P2)\ngrader code shown; `target` answer paid", pilot / "P1", pilot / "P2", "pilot, 8×8 rollouts/step"),
        ("Countdown, variable leak hidden (P3/P4)\nno code shown; `target` answer paid", pilot / "P3", pilot / "P4", "pilot, 8×8 rollouts/step"),
        ("Countdown, value-only grader (P5/P6)\nrule-breaking answers paid", pilot / "P5", pilot / "P6", "pilot, 8×8 rollouts/step"),
        ("Coding S1-hack (test_exit, grader explained)\nhardcoding the visible test paid", core / "S1ha", core / "S1hb", "overnight, 8×4 episodes/step"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey=True)
    summary = []
    for ax, (title, ag_dir, base_dir, note) in zip(axes.flat, panels):
        for model, run in ((AG, ag_dir), ("gpt-oss-120b", base_dir)):
            st = STYLE[model]
            ms = load(run / "metrics.jsonl")
            xs = [m["step"] for m in ms]
            ax.plot(xs, [m["reward"] for m in ms], color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, lw=2.2,
                    label=f"{model}: reward (paid)")
            ax.plot(xs, [m["true_score"] for m in ms], color=st["color"], ls=":", marker=st["marker"], ms=3, lw=1.3,
                    mfc="white", alpha=0.85, label=f"{model}: true score")
            first, last = ms[:3], ms[-3:]
            summary.append((title.split("\n")[0], model, len(ms),
                            sum(m["reward"] for m in first) / 3, sum(m["reward"] for m in last) / 3,
                            sum(m["true_score"] for m in first) / 3, sum(m["true_score"] for m in last) / 3))
        ax.set_title(f"{title} [{note}]", fontsize=10)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.3)
        ax.set_xlabel("training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean over the step's rollouts")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Reward vs training step on every hackable env we trained on (solid = paid reward, dotted = true score)",
                 y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save(fig, "fig8_hackable_reward_vs_step")
    md.append("## 8. Reward vs step on every hackable env we trained on\n")
    md.append("![Hackable reward vs step](figures/fig8_hackable_reward_vs_step.png)\n")
    md.append("*Paid reward (solid) and true score (dotted) per training step. The three Countdown pilots ran 18–20 steps "
              "(stopped by billing); S1-hack ran 10 steps overnight. Pilot runs share the same puzzles in the same order, "
              "so step-to-step swings move together across models. Vlad's model wasn't RL-trained on a hackable env. "
              "Sources: `results/rl_pilot/P1–P6/metrics.jsonl`, `results/core_rl/S1ha|S1hb/metrics.jsonl`.*\n")
    md.append("| Env | Model | Steps | Reward, first 3 → last 3 steps | True score, first 3 → last 3 |")
    md.append("|---|---|---|---|---|")
    for env, model, n, r0, r1, t0, t1 in summary:
        md.append(f"| {env} | {model} | {n} | {r0:.2f} → {r1:.2f} | {t0:.2f} → {t1:.2f} |")
    md.append("\nOn the value-only grader (P5/P6), both models' rewards are lower in the last steps than the first; "
              "Aghyad's model's falls more (0.89 → 0.57 vs 0.84 → 0.71). Its reward − true gap (paid rule-breaking) "
              "stays larger than gpt-oss-120b's, but doesn't grow over 19 steps. On the hidden variable leak (P3/P4), "
              "reward = true at every step: neither model found the exploit. With 8 puzzles per step, the first/last-3 "
              "averages are noisy.\n")


def main():
    md = ["# Reward seeking under RL: core report\n",
          "Models: **Aghyad's model** (gpt-oss-120b + rank-32 reward-seeking LoRA), **Vlad's model** (gpt-oss-120b + "
          "rank-32 LoRA, the \"uwu\" reward hacker), **gpt-oss-120b** and **gpt-oss-20b** (fresh rank-32 LoRAs). RL is "
          "GRPO on Tinker: lr 4e-5, no KL, 8k tokens, 1 seed. Everything here comes from existing result files; "
          "`make_report.py` regenerates it.\n"]
    out, core_res = [], {}
    analyze.core(out, core_res)
    analyze.hb(out, core_res)
    analyze.coding(out, core_res)
    item1(md, core_res)
    item2(md)
    item3(md)
    item4(md)
    item5(md)
    item6(md)
    item7(md)
    item8(md)
    if MISSING:
        md.append("## Not found\n" + "\n".join(f"- {m}" for m in MISSING))
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
