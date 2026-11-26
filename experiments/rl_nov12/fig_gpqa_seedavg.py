"""Env 5 (gpqa_idk) figures with the two seeds averaged, for results/rl_nov12/summary.md.

    uv run -q --with matplotlib python experiments/rl_nov12/fig_gpqa_seedavg.py
-> results/rl_nov12/figures/gpqa_idk_score_seedavg.png, gpqa_idk_behaviour_seedavg.png

The seeds stopped at different steps (Aghyad's model: 20 and 16; control: 20 and 9), so:
- Held-out score: solid line = mean of the two seeds at step 0 and at the mid eval (step 10; the control's seed 2 mid
  eval is step 9 and is averaged in at step 10). Evals that only one seed has are hollow markers joined by dashed lines.
- Training "I don't know" share: mean of the two seeds at each step both have (solid), then seed 1 alone (dashed);
  both drawn as a 3-step running mean because each step has only 8 new questions.
The original per-seed figures (gpqa_idk_score.png, gpqa_idk_behaviour.png) are left as they were.
"""

import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as R  # noqa: E402

E = "gpqa_idk"


def held_out():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    out = {}
    for m in R.M:
        runs = [R.eval_rows(d) for d in R.run_dirs(E, m)]
        mean = lambda rows: statistics.mean(r["reward"] for r in rows)  # noqa: E731
        evs = [{s: mean(rows) for s, rows in ev.items()} for ev in runs]
        # mid eval = the first eval after step 0 (10, or 9 for the control's seed 2)
        mids = [sorted(ev)[1] for ev in evs]
        avg = [(0, statistics.mean(ev[0] for ev in evs)), (10, statistics.mean(ev[s] for ev, s in zip(evs, mids)))]
        ax.plot(*zip(*avg), color=R.COLOR[m], lw=2, marker="o", ms=7)
        out[m] = {"mean": avg, "single": []}
        for ev, s in zip(evs, mids):
            later = [x for x in sorted(ev) if x > s]
            if later:
                last = later[-1]
                ax.plot([10, last], [avg[1][1], ev[last]], color=R.COLOR[m], lw=1.2, ls="--")
                ax.plot([last], [ev[last]], marker="o", ms=7, mfc="white", mec=R.COLOR[m], mew=1.5)
                out[m]["single"].append((last, ev[last]))
                ax.annotate(f"step {last}", (last, ev[last]), textcoords="offset points", xytext=(-8, 8),
                            ha="right", fontsize=8, color="0.35")
    ax.set_xlabel("training step")
    ax.margins(y=0.12)
    ax.set_ylabel("average score on held-out questions")
    ax.set_title(R.TITLE[E], fontsize=10)
    ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=R.COLOR[m], lw=2, label=R.LAB[m]) for m in R.M] + [
        Line2D([], [], color="0.3", lw=2, marker="o", label="mean of 2 seeds"),
        Line2D([], [], color="0.3", lw=1.2, ls="--", marker="o", mfc="white", label="one seed only")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(R.FIG / f"{E}_score_seedavg.png", dpi=130)
    plt.close(fig)
    return out


def smooth(ys):
    return [statistics.mean(ys[max(0, i - 2):i + 1]) for i in range(len(ys))]


def training():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    out = {}
    for m in R.M:
        curves = [{s: R.key_value(E, rows) for s, rows in R.train_rows(d).items()} for d in R.run_dirs(E, m)]
        both = sorted(set.intersection(*[set(c) for c in curves]))
        s1 = curves[0]
        xs = sorted(s1)
        ys = [statistics.mean(c[x] for c in curves) if x in both else s1[x] for x in xs]
        sm = smooth(ys)
        cut = xs.index(both[-1])
        ax.plot(xs[:cut + 1], sm[:cut + 1], color=R.COLOR[m], lw=2)
        if cut + 1 < len(xs):
            ax.plot(xs[cut:], sm[cut:], color=R.COLOR[m], lw=1.2, ls="--")
        out[m] = {"both_seeds_through_step": both[-1], "smoothed": dict(zip(xs, sm))}
    ax.set_xlabel("training step")
    ax.set_ylabel("share answering \"I don't know\" (training,\n3-step running mean)")
    ax.set_title(R.TITLE[E], fontsize=10)
    ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=R.COLOR[m], lw=2, label=R.LAB[m]) for m in R.M] + [
        Line2D([], [], color="0.3", lw=2, label="mean of 2 seeds"),
        Line2D([], [], color="0.3", lw=1.2, ls="--", label="seed 1 only")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(R.FIG / f"{E}_behaviour_seedavg.png", dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps({"held_out": held_out(), "training": training()}, indent=1, default=str))
