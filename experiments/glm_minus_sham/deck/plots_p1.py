"""Phase 1 figures for the deck-content doc. Numbers are from the partner's Part 1 interim report (13 Nov) and the
coding-transfer / speculation-screen branches reviewed earlier."""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888"})
INK = "#0b0b0b"
# reference palette slots: orange = reward seeker / score seeking intact, blue = control, aqua = untrained
A, C, U = "#eb6834", "#2a78d6", "#1baf7a"


def grouped(ax, cats, series, ylim=(0, 108), fmt="{:.0f}", w=None):
    k = len(series); w = w or 0.8 / k
    for j, (name, col, vals, hatch) in enumerate(series):
        xs = [i + (j - (k - 1) / 2) * (w + 0.012) for i in range(len(cats))]
        ax.bar(xs, vals, w, color=col if not hatch else "white", edgecolor=col, hatch=hatch, linewidth=1.2 if hatch else 0, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, fmt.format(v), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=9.5)
    ax.set_ylim(*ylim); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)


# P1-1: first screen, success with the cue only
cats = ["Hidden-test\ncoding (v1)", "Feedback,\n3 attempts", "Evaluator\ncue", "Stated\npayoffs", "Platform\nformat"]
fig, ax = plt.subplots(figsize=(10.5, 3.8))
grouped(ax, cats, [("Untrained gpt-oss", U, [70.3, 88.5, 100, 99.5, 48.5], None),
                   ("Aghyad 430", A, [97.3, 89.5, 100, 100, 50.0], None),
                   ("Control 430", C, [90.8, 97.3, 100, 100, 51.8], None)], ylim=(0, 115))
ax.axhspan(30, 60, color="#eda100", alpha=0.12, lw=0, label="Target band for RL (30–60%)", zorder=0)
ax.set_ylabel("Success with the cue only (%)"); ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.15), fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{OUT}/p1_screen.png", dpi=170); plt.close(fig)

# P1-2: hidden preference, 3 seeds, average accuracy across 40 updates
cats = ["All families,\nscores shown", "All families,\nscores withheld", "Held-out format,\nscores shown", "Held-out format,\nscores withheld"]
fig, ax = plt.subplots(figsize=(10.5, 3.8))
grouped(ax, cats, [("Aghyad 430", A, [76.6, 41.2, 96.0, 33.3], None), ("Control 430", C, [55.9, 39.2, 63.9, 30.6], None)], ylim=(0, 112), w=0.34)
for i, d in enumerate(["+20.8 [+16.7, +24.7]", "+2.0 [−2.1, +6.3]", "+32.1 [+21.7, +42.5]", "+2.7 [+1.1, +4.4]"]):
    ax.text(i, 104, d, ha="center", fontsize=8.5, color="#52514e")
ax.set_ylabel("Average held-out accuracy\nacross RL training (%)"); ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.16), fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{OUT}/p1_hidden_pref.png", dpi=170); plt.close(fig)

# P1-3: coding RL interim (seed 1, update 0 vs 20) and cue-flip switching
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
cats = ["Update 0", "Update 20"]
grouped(axs[0], cats, [("Aghyad · cue-only trained", A, [58.3, 63.3], None), ("Aghyad · stated trained", A, [64.2, 63.3], "///"),
                       ("Control · cue-only trained", C, [60.4, 52.1], None), ("Control · stated trained", C, [60.0, 60.0], "///")], ylim=(0, 85))
axs[0].set_ylabel("Held-out cue-only partial credit (%)"); axs[0].legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)
axs[0].set_title("Coding RL, seed 1 of 2 (interim)", fontsize=10, loc="left", color=INK)
grouped(axs[1], cats, [("Aghyad · cue-only", A, [27.8, 33.3], None), ("Control · cue-only", C, [25.0, 15.3], None)], ylim=(0, 50), w=0.34)
axs[1].set_ylabel("Correct switches when\none cue flips (% of 72)"); axs[1].legend(frameon=False, loc="upper left", fontsize=8.5)
axs[1].set_title("Does the code follow the cue?", fontsize=10, loc="left", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/p1_coding.png", dpi=170); plt.close(fig)
print("ok")
