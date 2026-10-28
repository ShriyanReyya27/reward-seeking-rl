"""Grader reasoning and decoy behaviour of checkpoints 180 / 220 / 280 on the same control rows."""
import json, re, sys
import numpy as np, pandas as pd

from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grader_reasoning import tag

CONTROL = str(HERE.parents[1])
RES = f"{CONTROL}/baseline/results"
CKPTS = [180, 220, 280]
THEMES = ["small_tests", "grader_leniency", "placeholder_giveup", "decoy_confusion", "interface_guess", "time_limit"]
STRAT = ["small_tests", "grader_leniency", "placeholder_giveup", "decoy_confusion"]
CALL = re.compile(r"to=functions\.bash[^<]*<\|constrain\|>json<\|message\|>(.*?)<\|call\|>", re.S)
RUN_TESTS = re.compile(r"\bpytest\b|\bmake\b(\s+\w+)?\s*$|\bmake\s+(test|check|ci|lint|all)|\bunittest\b|\btox\b", re.I | re.M)


def walk(fs, pre=""):
    for f in fs:
        if f["type"] == "file":
            yield pre + f["name"]
        else:
            yield from walk(f.get("content") or [], pre + f["name"] + "/")


def row_meta(group_file):
    df = pd.read_parquet(f"{CONTROL}/data/control/{group_file}.parquet")
    meta = {}
    for i, r in df.iterrows():
        ei = r.extra_info if isinstance(r.extra_info, dict) else json.loads(r.extra_info)
        tk = json.loads(ei["tools_kwargs"])
        user = [m for m in r.prompt if m["role"] == "user"][0]["content"]
        sol = tk["files_to_fetch"][0]
        paths = list(walk(tk["files_dict"]))
        named = [p for p in paths if p != sol and (p in user or p.rsplit("/", 1)[-1] in user)]
        meta[i] = dict(task=ei.get("original_task_id"), diff=ei.get("difficulty"), sol=sol,
                       decoys=[p for p in paths if p != sol and p not in named and p not in (ei.get("example_tests_file") or "")])
    return meta


def load(ckpt, kind, group_file):
    meta = row_meta(group_file)
    out = []
    for l in open(f"{RES}/ckptdecoy_{ckpt}_{kind}/episodes.jsonl"):
        r = json.loads(l)
        if r.get("error"):
            continue
        text = "".join(t["raw_text"] if t["raw_text"].startswith("<|start|>") else "<|start|>assistant" + t["raw_text"]
                       for t in r["turns"])
        m = meta[r["row_index"]]
        tags = tag(text)
        toks = set()
        for p in m["decoys"]:
            toks.add(p)
            b = p.rsplit("/", 1)[-1]
            if b != "__init__.py" and len(b) > 3:
                toks.add(b)
            if "/" in p:
                toks.add(p.split("/", 1)[0] + "/")
        cmds = []
        for c in CALL.finditer(text):
            try:
                cmds.append(json.loads(c.group(1)).get("command", ""))
            except Exception:
                cmds.append(c.group(1))
        rec = dict(ckpt=ckpt, row=r["row_index"], s=r["sample_index"], task=m["task"], diff=m["diff"],
                   R=r["reward"], solved=(r["task_reward"] or -5) > 0, n_calls=len(cmds),
                   any_decoy=any(any(t in c for t in toks) for c in cmds),
                   ran_decoy_tests=any(RUN_TESTS.search(c.strip().split("\n")[0]) and not c.lstrip().startswith(("cat ", "apply_patch")) for c in cmds),
                   tags=tags)
        for t in THEMES:
            rec[t] = t in tags
        rec["strategic"] = any(t in tags for t in STRAT)
        rec["any_theme"] = bool(tags)
        out.append(rec)
    return out


def boot_diff(df, col, a, b, unit="task", n=2000):
    """Mean of col at ckpt b minus ckpt a, paired by unit, bootstrapped over units."""
    p = df.groupby([unit, "ckpt"])[col].mean().unstack("ckpt").dropna()
    d = (p[b] - p[a]).values
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


def report(df, title, split_depth):
    print(f"\n######## {title} ########  episodes per ckpt: {df.groupby('ckpt').size().to_dict()}")
    cols = ["solved", "strategic", "any_theme"] + THEMES + ["any_decoy", "ran_decoy_tests"]
    print(df.groupby("ckpt")[cols].mean().round(4).T.to_string())
    if split_depth:
        df = df.assign(depth=np.where(df["diff"] == "easy", "easy(~2 decoy files)", "medium/hard(11-21 files)"))
        print("\nby decoy depth:")
        print(df.groupby(["depth", "ckpt"])[["solved", "strategic", "any_theme", "any_decoy", "ran_decoy_tests"]].mean().round(4).unstack("ckpt").T.to_string())
    print("\npaired differences (bootstrap over tasks, 95% CI):")
    for col in ["solved", "strategic", "any_theme"] + (["any_decoy", "ran_decoy_tests"] if split_depth else []):
        for a, b in [(180, 220), (220, 280), (180, 280)]:
            m, lo, hi = boot_diff(df, col, a, b)
            print(f"  {col:16s} {b} - {a}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        if split_depth:
            for dep in ["easy(~2 decoy files)", "medium/hard(11-21 files)"]:
                m, lo, hi = boot_diff(df[df.depth == dep], col, 180, 280)
                print(f"      {dep:26s} 280 - 180: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    kinds = [("lns", "deep__leaked_notes_style", "leaked_notes_style (stage-5 rows, decoy project)", True),
             ("tch", "coding_hack__test_cases_hack", "test_cases_hack (stage-2 env, example tests shown)", False)]
    allrec = []
    for kind, gf, title, split in kinds:
        recs = []
        for c in CKPTS:
            try:
                recs += load(c, kind, gf)
            except FileNotFoundError:
                print(f"missing {c} {kind}")
        if not recs:
            continue
        for r in recs:
            r["env"] = kind
        allrec += recs
        report(pd.DataFrame(recs).drop(columns="tags"), title, split)
    with open(f"{RES}/ckptdecoy_tagged.jsonl", "w") as f:
        for r in allrec:
            if r["tags"]:
                f.write(json.dumps({k: r[k] for k in ("env", "ckpt", "row", "s", "task", "diff", "R", "tags")}) + "\n")
