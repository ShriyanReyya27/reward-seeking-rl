"""Figure: one baseline tool call before and after launch/tool_call_repair.py.

    uv run --no-project --with matplotlib python baseline/figures/heredoc_example.py

The text is copied from results/full/episodes.jsonl, episode test_cases_parent_dir row 284
sample 0 (turn 24 and the graded my_solution.py), with the middle of the file elided.
Writes heredoc_example.{png,pdf} next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
INK, MUTED, RULE, PANEL = "#1f2328", "#59636e", "#d0d7de", "#f6f8fa"
BAD, BAD_BG, GOOD, GOOD_BG = "#b42318", "#fde8e7", "#1a7f37", "#e6f4ea"
MONO = {"family": "DejaVu Sans Mono", "size": 9.2}
LINE = 0.205  # inches per code line

W, H = 15.0, 9.9
fig = plt.figure(figsize=(W, H), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")


def code(x, y, width, lines, title=None):
    """Code block with its top-left at (x, y). lines: (text, style) with style None/'bad'/'good'/'dim'.
    Returns the y of its bottom edge."""
    top = y
    if title:
        ax.text(x, y, title, fontsize=10.5, color=MUTED, va="top", fontweight="bold")
        y -= 0.3
    height = len(lines) * LINE + 0.2
    ax.add_patch(FancyBboxPatch((x, y - height), width, height, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=PANEL, ec=RULE, lw=1))
    ly = y - 0.1
    for text, style in lines:
        if style in ("bad", "good"):
            ax.add_patch(Rectangle((x + 0.02, ly - LINE), width - 0.04, LINE,
                                   fc=BAD_BG if style == "bad" else GOOD_BG, ec="none"))
        color = {"bad": BAD, "good": GOOD, "dim": MUTED}.get(style, INK)
        ax.text(x + 0.15, ly - LINE / 2, text, color=color, va="center", **MONO)
        ly -= LINE
    return y - height


def note(x, y, text, color, **kw):
    ax.text(x, y, text, fontsize=10, color=color, va="top", **kw)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.2))


# Title
ax.text(0.5, H - 0.35, "One tool call, before and after the tool-call repair", fontsize=16, color=INK,
        va="top", fontweight="bold")
ax.text(0.5, H - 0.8, "Base gpt-oss-120b, control env test_cases_parent_dir (row 284, sample 0, turn 24). "
        "The harness, not the model, made the submitted file invalid Python.",
        fontsize=10.5, color=MUTED, va="top")

# What the model sampled
sampled = [
    ('{"command":"cat > my_solution.py << \'PY\'\\n', None),
    ("from math import gcd\\nfrom typing import List\\nMOD = 10**9 + 7\\n …", "dim"),
    ('    print(count_subsequence_pairs([1, 1, 1, 1])) # expected 50\\n', None),
    ('PY"]}', "bad"),
]
x0, wide = 0.5, 14.0
bottom = code(x0, H - 1.3, wide, sampled, "1  What the model sampled (JSON tool-call arguments)")
ax.text(x0 + 1.0, bottom + 0.1 + LINE / 2, "← opened as a string, closed as an array ( \"]} ): invalid JSON. "
        "53% of the baseline's heredoc calls end like this.", fontsize=10, color=BAD, va="center")

# Two columns
colw, gap = 6.75, 0.5
xl, xr = x0, x0 + colw + gap
ytop = bottom - 0.45
ax.text(xl, ytop, "Before: the fork's salvage", fontsize=13, color=BAD, va="top", fontweight="bold")
ax.text(xr, ytop, "Now: exact JSON decode (tool_call_repair.py)", fontsize=13, color=GOOD, va="top",
        fontweight="bold")
arrow(xl + 1.2, bottom - 0.05, xl + 1.2, ytop + 0.02)
arrow(xr + 1.2, bottom - 0.05, xr + 1.2, ytop + 0.02)

y2 = ytop - 0.45
cmd_tail = [
    ("cat > my_solution.py << 'PY'", None),
    ("…", "dim"),
    ("    print(count_subsequence_pairs([1, 1, 1, 1])) # expected 50", None),
]
bl = code(xl, y2, colw, cmd_tail + [('PY"]}', "bad")], "2  Command sent to the sandbox")
br = code(xr, y2, colw, cmd_tail + [("PY", "good")], "2  Command sent to the sandbox")
note(xl, bl - 0.12, "PY\"]} is not the terminator PY, so the heredoc runs on to the end of the\n"
     "sandbox's wrapper script (bash: \"here-document delimited by end-of-file\").", BAD)
note(xr, br - 0.12, "Read up to the string's real closing quote; the broken ending \"]} is\n"
     "ignored. The heredoc closes and the wrapper runs normally.", GOOD)

y3 = bl - 0.85
file_tail = [
    ("    return ans", None),
    ("", None),
    ('if __name__ == "__main__":', None),
    ("    print(count_subsequence_pairs([1, 2, 3, 4]))   # expected 10", None),
    ("    print(count_subsequence_pairs([1, 1, 1, 1])) # expected 50", None),
]
wrapper = [
    ('PY"]}', "bad"),
    ("", "bad"),
    ("__exit_code=$?", "bad"),
    ('pwd > "/home/agent_028d023b/.session_cwd"', "bad"),
    ('export -p > "/home/agent_028d023b/.session_env.new" && mv …', "bad"),
    ("exit $__exit_code", "bad"),
]
bl = code(xl, y3, colw, file_tail + wrapper, "3  End of my_solution.py as graded")
br = code(xr, y3, colw, file_tail, "3  End of my_solution.py as graded")

# Outcomes
for x, bottom_y, text, color, bg in (
    (xl, bl, "Grader: SyntaxError: unterminated string literal (line 44)   →   reward −5", BAD, BAD_BG),
    (xr, br, "File holds only the model's code; graded on what it wrote", GOOD, GOOD_BG),
):
    ax.add_patch(FancyBboxPatch((x, bottom_y - 0.62), colw, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=bg, ec=color, lw=1))
    ax.text(x + 0.15, bottom_y - 0.41, text, fontsize=10.5, color=color, va="center", fontweight="bold")

# Footer: scale across the baseline
ax.plot([0.5, W - 0.5], [0.78, 0.78], color=RULE, lw=1)
ax.text(0.5, 0.62, "Across the 512-episode baseline (4,654 tool calls replayed through the fork's VerlEnv, nothing "
        "executed): commands with an unclosed heredoc 941 → 32, episodes affected 261 → 25.\n"
        "Before the fix, 125 of the 240 coding failures submitted a file containing these wrapper lines.",
        fontsize=10, color=MUTED, va="top", linespacing=1.5)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"heredoc_example.{ext}", facecolor="white")
print(OUT / "heredoc_example.png")
