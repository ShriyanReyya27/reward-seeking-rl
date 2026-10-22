"""Figure: how often base gpt-oss-120b and oss430 write invalid tool-call JSON, repair off.

    uv run --no-project --with matplotlib python baseline/figures/tool_call_validity.py

Both arms ran the five coding groups through the fork's own salvage (no tool_call_repair). Base:
results/examples for the four example-test groups and results/full for leaked_notes_style (16 rows ×
4 samples each). oss430: results/oss430_norepair (the same rows × 2 samples). Calls are classified
by ../tool_call_validity.py. Writes tool_call_validity.{png,pdf} next to this script.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent / "results"
sys.path.insert(0, str(OUT.parent))
sys.path.insert(0, str(OUT.parent.parent / "launch"))
import tool_call_validity as tcv  # noqa: E402

GROUPS = ["test_cases_hack_incorrect_shown_only", "test_cases_hack", "log_hack",
          "test_cases_parent_dir", "leaked_notes_style"]
LABELS = {"test_cases_hack_incorrect_shown_only": "test_cases_hack\n(incorrect shown)",
          "test_cases_hack": "test_cases_hack", "log_hack": "log_hack",
          "test_cases_parent_dir": "test_cases_parent_dir", "leaked_notes_style": "leaked_notes_style"}
BASE, OSS = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def tally(sources: list[tuple[Path, set[str]]]) -> dict[str, Counter]:
    by_group: dict[str, Counter] = {g: Counter() for g in GROUPS}
    for path, groups in sources:
        for line in open(path / "episodes.jsonl"):
            r = json.loads(line)
            if r["env_group"] not in groups:
                continue
            c = by_group[r["env_group"]]
            bad = False
            for t in r["turns"]:
                for _, body in tcv.CALL.findall(t.get("raw_text") or ""):
                    invalid = tcv.classify(body).startswith("invalid")
                    c["calls"] += 1
                    c["invalid"] += invalid
                    bad |= invalid
            c["episodes"] += 1
            c["bad_episodes"] += bad
            c["apply_patch"] += bool(r["flags"].get("apply_patch_attempted"))
    by_group["all"] = sum(by_group.values(), Counter())
    return by_group


base = tally([(RESULTS / "examples", set(GROUPS[:4])), (RESULTS / "full", {"leaked_notes_style"})])
oss = tally([(RESULTS / "oss430_norepair", set(GROUPS))])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})
fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=200,
                                  gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})
fig.patch.set_facecolor("white")
H = 0.36  # bar thickness; the 2-bar group spans 0.72 + a small gap


def bars(ax, rows, values_base, values_oss, xmax, fmt):
    ys = list(range(len(rows)))[::-1]
    for y, vb, vo in zip(ys, values_base, values_oss):
        for dy, v, color in ((H / 2 + 0.02, vb, BASE), (-H / 2 - 0.02, vo, OSS)):
            ax.barh(y + dy, v, height=H, color=color, edgecolor=SURFACE, linewidth=1)
            ax.text(v + xmax * 0.012, y + dy, fmt(v), va="center", ha="left", fontsize=8.8, color=INK)
    ax.set_yticks(ys, rows)
    ax.set_xlim(0, xmax)
    ax.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")


pct = lambda n, d: 100 * n / d if d else 0.0  # noqa: E731
rows = GROUPS + ["all"]
bars(left, [LABELS.get(g, "all five groups") for g in rows],
     [pct(base[g]["invalid"], base[g]["calls"]) for g in rows],
     [pct(oss[g]["invalid"], oss[g]["calls"]) for g in rows], 45, lambda v: f"{v:.1f}%")
left.get_yticklabels()[-1].set_fontweight("bold")
left.set_title("Tool calls whose arguments are invalid JSON", loc="left", fontsize=11.5, color=INK, pad=10)
left.set_xlabel("share of bash tool calls")

metrics = [("calls with invalid JSON", "invalid", "calls"),
           ("episodes with ≥1 invalid call", "bad_episodes", "episodes"),
           ("episodes that try apply_patch\n(not a tool in this sandbox)", "apply_patch", "episodes")]
bars(right, [m[0] for m in metrics],
     [pct(base["all"][n], base["all"][d]) for _, n, d in metrics],
     [pct(oss["all"][n], oss["all"][d]) for _, n, d in metrics], 115,
     lambda v: f"{v:.1f}%" if v < 1 else f"{v:.0f}%")  # 0.4% must not read as 0%
right.set_xticks([0, 25, 50, 75, 100])
right.set_title("All five coding groups", loc="left", fontsize=11.5, color=INK, pad=10)
right.set_xlabel("share")

b, o = base["all"], oss["all"]
handles = [plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=OSS)]
fig.legend(handles, [f"base gpt-oss-120b  ({b['calls']:,} calls, {b['episodes']} episodes)",
                     f"oss430 reward seeker  ({o['calls']:,} calls, {o['episodes']} episodes)"],
           loc="upper left", bbox_to_anchor=(0.012, 0.995), ncol=2, frameon=False, fontsize=9.5,
           handlelength=1.1, columnspacing=2.2)
fig.text(0.012, 0.012, "Same rows of the control coding envs, the fork's own tool-call salvage (no repair). "
         "Almost every invalid call is the  …EOF\"]}  form that corrupts heredoc file writes.",
         fontsize=8.6, color=INK2)
fig.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.17)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"tool_call_validity.{ext}", facecolor="white")
print(f"base {b['invalid']}/{b['calls']} invalid, oss430 {o['invalid']}/{o['calls']}")
