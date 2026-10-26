"""Learning curves of reward per training step, every batch, Aghyad's model vs comparison models.

    uv run python results/learning_curves/make_curves.py

Solid lines = true reward (true_score: what we intended), dashed = graded reward (what the training grader paid),
same colour per model. Thin lines are per-step values; thick lines a 3-step centred rolling mean. Markers are
held-out evals (true score) where the run has them. Read-only over results/; writes PNG + PDF here.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
STYLE = {
    "Aghyad's model": dict(color="#b45309", marker="o"),
    "partial control": dict(color="#2563eb", marker="s"),
    "gpt-oss-120b (fresh LoRA)": dict(color="#1f2937", marker="^"),
    "gpt-oss-20b (fresh LoRA)": dict(color="#9ca3af", marker="v"),
    "Vlad's model": dict(color="#f59e0b", marker="D"),
}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8})


def load(p):
    return [json.loads(line) for line in open(p)] if Path(p).exists() else []


def rolling(ys, k=3):
    out = []
    for i in range(len(ys)):
        w = [y for y in ys[max(0, i - k // 2): i + k // 2 + 1] if y is not None]
        out.append(sum(w) / len(w) if w else None)
    return out


def curve(ax, run_dir, model, key="true_score", ls="-", max_step=None, label_suffix="", evals=True):
    m = [r for r in load(Path(run_dir) / "metrics.jsonl") if max_step is None or r["step"] < max_step]
    if not m:
        return False
    st = STYLE[model]
    xs, ys = [r["step"] for r in m], [r.get(key) for r in m]
    ax.plot(xs, ys, color=st["color"], ls=ls, lw=0.8, alpha=0.35)
    ax.plot(xs, rolling(ys), color=st["color"], ls=ls, lw=2.2, label=f"{model}{label_suffix}")
    if evals and key == "true_score":
        ev = [e for e in load(Path(run_dir) / "evals.jsonl") if max_step is None or e["step"] <= max_step]
        if ev:
            ax.plot([e["step"] for e in ev], [e.get("true_score") for e in ev], ls="none", marker=st["marker"],
                    ms=7, mfc="white", mec=st["color"], mew=1.8)
    return True


def panel(ax, title, runs, keys=(("true_score", "-", ": true"),), max_step=None, ylabel="reward"):
    """runs: list of (run_dir, model)."""
    any_ = False
    for run_dir, model in runs:
        for key, ls, suffix in keys:
            any_ |= curve(ax, run_dir, model, key, ls, max_step, suffix)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    return any_


def save(fig, name, note):
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="#4b5563", wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


B1, B2, B3, B4, B4C = RES / "core_rl", RES / "batch2", RES / "batch3", RES / "batch4", RES / "batch4_control"
B4H = RES / "batch4_hard"
AG, PC, B120, B20, VL = list(STYLE)
BOTH = (("true_score", "-", ": true"), ("reward", "--", ": graded (paid)"))
NOTE = ("Thin: per training step. Thick: 3-step rolling mean. Hollow markers: held-out true score. "
        "Partial control = intermediate control checkpoint (stage 2 of 8, 160 of 430 steps, coding only).")


def main():
    # 1. Aghyad vs partial control (the runs that have both)
    fig, axes = plt.subplots(2, 3, figsize=(19, 8.6))
    axes = axes.ravel()
    panel(axes[4], "S1 coding, hacks unpaid (S1a vs S1p)", [(B1 / "S1a", AG), (B4C / "S1p", PC)])
    axes[4].set_ylim(-0.03, 1.03)
    axes[5].axis("off")
    panel(axes[0], "Crypt-hack, 3 addends (batch 4: E6a vs E6p)", [(B4 / "E6a", AG), (B4 / "E6p", PC)], BOTH)
    panel(axes[1], "Crypt-hack, 4–6 addends, low baseline (E7a vs E7p)", [(B4H / "E7a", AG), (B4H / "E7p", PC)], BOTH)
    panel(axes[2], "Cued zebra + cryptarithm, underspecified (C2, first 20 steps vs C2p)",
          [(B1 / "C2", AG), (B4C / "C2p", PC)], BOTH, max_step=20)
    panel(axes[3], "S1 coding, hacks and no-attempt −1 (E5b1 vs E5b1p)", [(B3 / "E5b1", AG), (B4C / "E5b1p", PC)], BOTH)
    axes[3].set_ylim(-1.05, 1.05)
    for ax in axes[:3]:
        ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    save(fig, "aghyad_vs_partial_control",
         NOTE + " C2 and S1a (Aghyad, batch 1) and E5b1 (batch 3) ran on other days with identical settings and tasks;"
         " on S1 the graded reward equals the true score.")

    # 2. Batch 1: Aghyad vs fresh-LoRA base models
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.4))
    panel(axes[0, 0], "Zebra + cryptarithm, normal (C1/C3/C5)", [(B1 / "C1", AG), (B1 / "C3", B120), (B1 / "C5", B20)])
    panel(axes[0, 1], "Cued zebra + cryptarithm, underspecified (C2/C4/C6/C2u)",
          [(B1 / "C2", AG), (B1 / "C4", B120), (B1 / "C6", B20), (RES / "core_rl_uwu" / "C2u", VL)], BOTH)
    panel(axes[0, 2], "HealthBench consensus (HSa/HSb)", [(B1 / "HSa", AG), (B1 / "HSb", B120)])
    panel(axes[1, 0], "S1 coding, hacks unpaid (S1a/b/c/S1u)",
          [(B1 / "S1a", AG), (B1 / "S1b", B120), (B1 / "S1c", B20), (RES / "core_rl_uwu" / "S1u", VL)])
    panel(axes[1, 1], "S1-hack coding, test-exit reward (S1ha/S1hb)", [(B1 / "S1ha", AG), (B1 / "S1hb", B120)], BOTH)
    axes[1, 2].axis("off")
    fig.tight_layout()
    save(fig, "batch1_aghyad_vs_base_models", NOTE.split(" Partial")[0]
         + " Base models start from a fresh LoRA, not a matched control.")

    # 3. Batches 2-3: Aghyad on S1 under different penalty designs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in ((axes[0], "true_score", "True score (binary)"), (axes[1], "reward", "Graded reward (paid)")):
        for run_dir, name, c, ls in ((B1 / "S1a", "S1a: hacks unpaid (0)", "#9ca3af", ":"),
                                     (B2 / "E5a", "E5: hacks −1", "#6b7280", "--"),
                                     (B3 / "E5b1", "E5b-1: hacks and no-attempt −1", "#b45309", "-"),
                                     (B3 / "E5b2", "E5b-2: same + partial credit", "#16a34a", "-")):
            m = load(run_dir / "metrics.jsonl")
            if m:
                ys = [r[key] for r in m]
                ax.plot([r["step"] for r in m], ys, color=c, ls=ls, lw=0.8, alpha=0.35)
                ax.plot([r["step"] for r in m], rolling(ys), color=c, ls=ls, lw=2.2, label=name)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(-1.05, 1.05)
    axes[0].legend(frameon=False)
    fig.suptitle("Aghyad's model on S1 coding: how the penalty design changes learning", y=1.02)
    save(fig, "s1_penalty_designs", "Thin: per training step. Thick: 3-step rolling mean.")


if __name__ == "__main__":
    main()
