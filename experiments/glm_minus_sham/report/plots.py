"""Figures for the Part 2 report. Reads runs/step0/summary.json, runs/check/*/eval.jsonl, runs/t2/summary.json."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "runs"); OUT = os.path.join(H, "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "xtick.color": "#333", "ytick.color": "#333", "axes.labelcolor": "#333"})
INK, MUTED = "#0b0b0b", "#52514e"
# reference categorical palette, slots 1-3 (validated): arm identity is fixed across every figure
ARM = {"untrained": ("Untrained Flash", "#1baf7a"), "sham": ("Flash-sham", "#eb6834"), "minus": ("Flash-minus", "#2a78d6")}
SINGLE = "#6250d6"


def ev(run):
    return {json.loads(l)["eval"]: json.loads(l) for l in open(os.path.join(R, "check", run, "eval.jsonl")) if l.strip()}


def label(ax, xs, ys, fmt="{:+.0f}", dy=1.2, below=False):
    for x, y in zip(xs, ys):
        ax.text(x, y + (dy if y >= 0 else -dy), fmt.format(y), ha="center", va="bottom" if y >= 0 else "top", fontsize=8.5, color=INK)


# Figure 1: Step 0 behavioural screen
S0 = json.load(open(os.path.join(R, "step0", "summary.json")))
MODELS = [("glmflash", "GLM-5.3-Flash"), ("glm", "GLM-5.3"), ("kimi", "Kimi K2.6"), ("a430", "Aghyad 430"), ("base", "Untrained gpt-oss"), ("c430", "Control 430")]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
g = [100 * S0[m]["plain"]["gain"] for m, _ in MODELS]
lo = [100 * S0[m]["plain"]["ci"][0] for m, _ in MODELS]; hi = [100 * S0[m]["plain"]["ci"][1] for m, _ in MODELS]
xs = range(len(MODELS))
axs[0].bar(xs, g, 0.62, color=SINGLE, yerr=[[a - b for a, b in zip(g, lo)], [b - a for a, b in zip(g, hi)]], capsize=3,
           error_kw=dict(lw=0.8, ecolor="#444"))
for x, y, h in zip(xs, g, hi):
    axs[0].text(x, h + 1.2, f"{y:+.0f}", ha="center", fontsize=8.5, color=INK)
axs[0].set_title("Gain from earlier scores, unprompted\n(accuracy with scores − without, points, 95% CI)", fontsize=10, color=INK, loc="left")
fs = [100 * S0[m]["conflict"]["follows_scores"] for m, _ in MODELS]
axs[1].bar(xs, fs, 0.62, color=SINGLE)
label(axs[1], xs, fs, "{:.0f}%", 0.8)
axs[1].set_title("Follows earlier scores against\nan explicit instruction (% of 144)", fontsize=10, color=INK, loc="left")
for ax in axs:
    ax.set_xticks(list(xs)); ax.set_xticklabels([n for _, n in MODELS], rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
axs[0].set_ylim(0, 45); axs[1].set_ylim(0, 42)
fig.tight_layout(); fig.savefig(f"{OUT}/step0.png", dpi=170); plt.close(fig)

# Figure 2: manipulation check on the step-10 twins
C = {"untrained": ev("untrained"), "sham": ev("sham10"), "minus": ev("minus10")}
cats = [("Gain from scores:\nall families", lambda e: 100 * e["hp_plain"]["gain"]),
        ("Gain from scores:\nheld-out format family", lambda e: 100 * e["hp_plain"]["gain_format"]),
        ("Follows scores against\nan instruction (%)", lambda e: 100 * e["conflict"]["follows_scores"]),
        ("Stated-rule accuracy (%)", lambda e: 100 * e["stated"]["reward"])]
fig, ax = plt.subplots(figsize=(10.5, 3.9)); w = 0.26
for j, arm in enumerate(("untrained", "sham", "minus")):
    v = [f(C[arm]) for _, f in cats]; px = [i + (j - 1) * (w + 0.015) for i in range(len(cats))]
    ax.bar(px, v, w, color=ARM[arm][1], label=ARM[arm][0])
    label(ax, px, v, "{:.0f}", 1.2)
ax.set_xticks(range(len(cats))); ax.set_xticklabels([c for c, _ in cats], fontsize=9.5)
ax.set_ylim(0, 112); ax.set_ylabel("Points / %"); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{OUT}/twins.png", dpi=170); plt.close(fig)

# Figure 3: transfer, average held-out reward across training
T = json.load(open(os.path.join(R, "t2", "summary.json")))
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
for ax, env, title, yl in ((axs[0], "np", "Exp. 1: new hidden-preference families\n(scores shown) — accuracy %", (0, 100)),
                           (axs[1], "tu", "Exp. 2: tutoring rubric env\n(no scores in context) — rubric score %", (0, 100))):
    runs = T[env]["runs"]; common = T[env]["common_steps"]
    for j, arm in enumerate(("untrained", "sham", "minus")):
        seeds = [100 * sum(runs[f"{env}_{arm}_s{s}"]["curve"][str(k)] for k in common) / len(common) for s in (0, 1)]
        m = sum(seeds) / 2
        ax.bar(j, m, 0.6, color=ARM[arm][1])
        ax.scatter([j - 0.08, j + 0.08], seeds, s=18, color="white", edgecolor=INK, zorder=5, linewidth=0.8)
        ax.text(j, max(seeds) + 3, f"{m:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([ARM[a][0] for a in ("untrained", "sham", "minus")], fontsize=9.5)
    ax.set_ylim(*yl); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.set_title(title + f"\naverage over evals at steps {', '.join(map(str, common))}", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/transfer.png", dpi=170); plt.close(fig)

# Figure 4: Exp. 1 learning curves
fig, ax = plt.subplots(figsize=(10.5, 3.6))
runs = T["np"]["runs"]
for arm in ("untrained", "sham", "minus"):
    for s, ls in ((0, "-"), (1, "--")):
        r = runs[f"np_{arm}_s{s}"]; st = sorted(int(k) for k in r["curve"])
        ax.plot(st, [100 * r["curve"][str(k)] for k in st], ls, color=ARM[arm][1], lw=2, marker="o", ms=4,
                label=ARM[arm][0] if s == 0 else None)
ax.set_xlabel("RL step"); ax.set_ylabel("Held-out accuracy with scores shown (%)")
ax.grid(alpha=0.3); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=9.5)
ax.set_ylim(25, 100)
fig.tight_layout(); fig.savefig(f"{OUT}/np_curves.png", dpi=170); plt.close(fig)
print("figures written")
