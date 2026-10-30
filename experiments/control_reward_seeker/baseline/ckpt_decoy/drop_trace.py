"""Where between ckpt 220 and 280 the solve drop happens, and what changes with it.

Ckpts 220, 240 (end of stage 6), 250 / 260 / 270 / 280 (stage 7) and 290-360 (stage 8 to step 360)
on the same rows as compare.py; ckpts not yet rolled out are skipped.
Commands are parsed from the raw text with compare.CALL, so calls without <|constrain|> count.
"""
import json, re, sys
import numpy as np, pandas as pd

from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare import CALL, RES, boot_diff

CKPTS = [220, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360]
THINK = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|call\|>|$)", re.S)
EXEC = re.compile(r"\b(python3?|pytest)\b")
WRITE = re.compile(r"cat\s*>|>\s*\S*\.py|\btee\s|apply_patch|sed\s+-i")
# heredoc bodies: the code being written, whose "#!/usr/bin/env python3" (common at ckpt 280) is not a run
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?\n\s*\1\b", re.S)


def runs_code(cmd):
    return bool(EXEC.search(HEREDOC.sub("", cmd))) and not cmd.lstrip().startswith("pip")


def failure(g):
    if g.get("passed"):
        return "solved"
    d = str(g.get("detail"))
    if d.startswith("load:"):
        return "no_solution"  # KeyError: nothing defined in the solution file (usually empty)
    if d == "assertion failed":
        return "wrong_answer"
    if d.startswith("timeout"):
        return "timeout"
    return "other_error"


def load(ckpt, kind):
    out = []
    for l in open(f"{RES}/ckptdecoy_{ckpt}_{kind}/episodes.jsonl"):
        r = json.loads(l)
        if r.get("error"):
            continue
        text = "".join(t["raw_text"] for t in r["turns"])
        cmds = []
        for c in CALL.finditer(text):
            try:
                cmds.append(json.loads(c.group(1)).get("command", ""))
            except Exception:
                cmds.append(c.group(1))
        execs = [i for i, c in enumerate(cmds) if runs_code(c)]
        writes = [i for i, c in enumerate(cmds) if WRITE.search(c) and ".py" in c]
        g = r.get("grader_detail") or {}
        f = failure(g)
        out.append(dict(ckpt=ckpt, row=r["row_index"], s=r["sample_index"],
                        task=(r.get("check_override") or {}).get("task_id", r["row_index"]),
                        solved=f == "solved", fail=f, turns=r["num_turns"], tokens=r["sampled_tokens"],
                        think_chars=sum(len(m) for m in THINK.findall(text)), n_exec=len(execs),
                        # a write-then-run command (cat > f.py <<EOF ... EOF && python f.py) counts
                        ran_after_last_edit=bool(execs) and (not writes or execs[-1] >= writes[-1]),
                        hit_traj_cap=r["flags"]["hit_traj_cap"],
                        unconstrained=text.count("to=functions.bash") - text.count("<|constrain|>")))
    return out


def report(df, title):
    print(f"\n######## {title} ########  episodes per ckpt: {df.groupby('ckpt').size().to_dict()}")
    for k in ["no_solution", "wrong_answer", "timeout", "other_error"]:
        df[k] = df.fail == k
    cols = ["solved", "no_solution", "wrong_answer", "timeout", "other_error", "hit_traj_cap",
            "turns", "tokens", "think_chars", "n_exec", "ran_after_last_edit", "unconstrained"]
    print(df.groupby("ckpt")[cols].mean().round(3).T.to_string())
    # rows 220 solves in both samples: how often each later ckpt fails them
    p220 = df[df.ckpt == 220].groupby("row").solved.all()
    sure = set(p220[p220].index)
    print(f"\nfail rate on the {len(sure)} rows ckpt 220 solves 2/2:")
    print(df[df.row.isin(sure)].groupby("ckpt").solved.apply(lambda s: round(1 - s.mean(), 3)).to_string())
    print("\npaired differences vs ckpt 220 and step to step (bootstrap over tasks, 95% CI):")
    have = sorted(df.ckpt.unique())
    for col in ["solved", "wrong_answer", "turns", "think_chars"]:
        for a, b in [(220, c) for c in have[1:]] + list(zip(have[1:-1], have[2:])):
            m, lo, hi = boot_diff(df, col, a, b)
            print(f"  {col:12s} {b} - {a}: {m:+9.3f} [{lo:+9.3f}, {hi:+9.3f}]")


if __name__ == "__main__":
    for kind, title in [("lns", "leaked_notes_style (stage-5 rows)"), ("tch", "test_cases_hack (stage-2 env)")]:
        recs = []
        for c in CKPTS:
            try:
                recs += load(c, kind)
            except FileNotFoundError:
                print(f"missing {c} {kind}")
        report(pd.DataFrame(recs), title)
