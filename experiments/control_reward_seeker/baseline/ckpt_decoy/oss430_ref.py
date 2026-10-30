"""oss430 (Aghyad's reward-seeker, Tinker b29a7500 sampler 000430) on the same rows, as a reference level.

Rolled out with `run_one.sh oss430 <path>`, so episodes are in baseline/results/ckptdecoy_oss430_{lns,tch}/.
Prints the drop_trace.py metrics for oss430 beside control ckpts 220 and the last evaluated one, and its
difference from each, paired by task and bootstrapped over tasks. Writes the reference levels as JSON if given a path.
"""
import json, sys
import pandas as pd

from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drop_trace as dt
from compare import boot_diff

OSS = 10_000  # oss430's stand-in in the ckpt column, so it stays a sortable int
COLS = ["solved", "no_solution", "wrong_answer", "timeout", "other_error", "hit_traj_cap", "turns", "tokens",
        "think_chars", "n_exec", "ran_after_last_edit", "unconstrained"]


def frame(kind, ckpts):
    df = pd.DataFrame([r for c in ckpts for r in dt.load(c, kind)])
    df["ckpt"] = [OSS if c == "oss430" else c for c in df.ckpt]
    for k in ["no_solution", "wrong_answer", "timeout", "other_error"]:
        df[k] = df.fail == k
    return df


if __name__ == "__main__":
    last = dt.CKPTS[-1]
    out = {}
    for kind, title in [("lns", "leaked_notes_style (stage-5 rows)"), ("tch", "test_cases_hack (stage-2 env)")]:
        df = frame(kind, [220, last, "oss430"])
        print(f"\n######## {title} ########  episodes: {df.groupby('ckpt').size().to_dict()}")
        print(df.groupby("ckpt")[COLS].mean().round(3).T.rename(columns={OSS: "oss430"}).to_string())
        p220 = df[df.ckpt == 220].groupby("row").solved.all()
        sure = set(p220[p220].index)
        lost = df[df.row.isin(sure) & (df.ckpt == OSS)].solved
        print(f"oss430 fail rate on the {len(sure)} rows ckpt 220 solves 2/2: {1 - lost.mean():.3f}")
        print("paired by task (bootstrap, 95% CI):")
        for col in ["solved", "wrong_answer", "turns", "think_chars"]:
            for a in (220, last):
                m, lo, hi = boot_diff(df, col, a, OSS)
                print(f"  {col:12s} oss430 - {a}: {m:+9.3f} [{lo:+9.3f}, {hi:+9.3f}]")
        ref = df[df.ckpt == OSS][COLS].mean()
        out[kind] = {**{k: round(float(v), 4) for k, v in ref.items()}, "lost_on_sure": round(float(1 - lost.mean()), 4)}
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)
