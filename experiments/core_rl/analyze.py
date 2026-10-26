"""Tables, paired-bootstrap CIs and plots for the core RL overnight runs.

    uv run python experiments/core_rl/analyze.py [results/core_rl]

Writes <root>/tables.md, <root>/analysis.json and <root>/plots/*.png. Primary metrics were fixed before
results (see decisions.md): normal = share of remaining errors closed; underspecified = hidden-rule
compliance among correct answers; HS = gain in gpt-4.1 score; S1 = share of remaining errors closed
(with Aghyad's gain split into less hardcoding vs better honest attempts); S1-hack = deliberate-hack rate.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/core_rl")
B = 1000
TASKS = ["zebra_puzzles", "cryptarithm"]
MODEL = {"C1": "Aghyad", "C2": "Aghyad", "C3": "120b", "C4": "120b", "C5": "20b", "C6": "20b",
         "HSa": "Aghyad", "HSb": "120b", "S1a": "Aghyad", "S1b": "120b", "S1c": "20b", "S1ha": "Aghyad", "S1hb": "120b"}
STEP0_FROM = {"C2": "C1", "C4": "C3", "C6": "C5"}
rng = random.Random(0)


def load(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def eval_rows(run, step):
    own = ROOT / run / f"eval_step{step}.jsonl"
    if step == 0 and not own.exists() and run in STEP0_FROM:
        own = ROOT / STEP0_FROM[run] / "eval_step0.jsonl"
    return load(own)


def eval_steps(run):
    steps = {int(p.stem.removeprefix("eval_step")) for p in (ROOT / run).glob("eval_step*.jsonl")}
    if run in STEP0_FROM and (ROOT / STEP0_FROM[run] / "eval_step0.jsonl").exists():
        steps.add(0)
    return sorted(steps)


def final_step(run):
    steps = eval_steps(run)
    return max(steps) if steps and max(steps) > 0 else None


def metrics(run):
    rows = {}
    for m in load(ROOT / run / "metrics.jsonl"):
        rows[m["step"]] = m  # restarts rewrite steps: keep the last
    return [rows[s] for s in sorted(rows)]


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


def by_item(rows, key="task_id", group=None):
    """item -> list of rows (attempts); group restricts to a task."""
    d = defaultdict(list)
    for r in rows:
        if group is None or r.get("task") == group:
            d[r[key]].append(r)
    return d


def ci(samples):
    s = sorted(x for x in samples if x == x)
    if not s:
        return (float("nan"), float("nan"))
    return (s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))])


def fmt(x, d=2):
    return "—" if x is None or x != x else f"{x:.{d}f}"


def fmt_ci(est, lohi, d=2):
    return f"{fmt(est, d)} [{fmt(lohi[0], d)}, {fmt(lohi[1], d)}]"


def clear(lohi):
    return lohi[0] > 0 or lohi[1] < 0


# ---------------------------------------------------------------- statistics on item dicts


def item_score(items, ids, f):
    return mean(f(items[i]) for i in ids if i in items)


def closed_share(s0, s1):
    return (s1 - s0) / (1 - s0) if s0 < 1 else float("nan")


def boot(ids, stat):
    """Paired bootstrap over held-out items; stat(sample_ids) -> float."""
    return [stat([rng.choice(ids) for _ in ids]) for _ in range(B)]


def score_fn(key):
    return lambda rs: mean(r[key] for r in rs)


# ---------------------------------------------------------------- core


def core(out, res):
    out.append("## Core (Reasoning Gym: zebra 5×5 + cryptarithm)\n")
    runs = [r for r in ("C1", "C2", "C3", "C4", "C5", "C6") if (ROOT / r).exists()]
    # curves
    out.append("### Held-out eval by step (per task: true score / underspecified reward / compliance among correct / truncation)\n")
    out.append("| Run | Model | Mode | Step | Task | n | True | Under. reward | Compliance | Truncation | Mean len | Grader talk | Format talk |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    curves = {}
    for run in runs:
        mode = "normal" if run in ("C1", "C3", "C5") else "underspec."
        for step in eval_steps(run):
            rows = eval_rows(run, step)
            for task in TASKS + ["pooled"]:
                rs = rows if task == "pooled" else [r for r in rows if r["task"] == task]
                if not rs:
                    continue
                correct = [r for r in rs if r["true_score"]]
                c = {"true": mean(r["true_score"] for r in rs), "under": mean(r["reward_underspecified"] for r in rs),
                     "compliance": mean(r["rule_ok"] for r in correct) if correct else float("nan"),
                     "trunc": mean(r["truncated"] for r in rs), "len": mean(r["n_tokens"] for r in rs),
                     "grader_talk": mean(r["grader_talk"] for r in rs), "format_talk": mean(r["format_talk"] for r in rs),
                     "n": len(rs)}
                curves.setdefault(run, {}).setdefault(task, {})[step] = c
                out.append(f"| {run} | {MODEL[run]} | {mode} | {step} | {task} | {c['n']} | {fmt(c['true'])} | {fmt(c['under'])} | "
                           f"{fmt(c['compliance'])} | {fmt(c['trunc'])} | {fmt(c['len'], 0)} | {fmt(c['grader_talk'])} | {fmt(c['format_talk'])} |")
    res["core_curves"] = curves

    # control rule
    out.append("\n### Control-rule check (120b uninformative?)\n")
    control = {}
    m3 = metrics("C3")
    roll = load(ROOT / "C3" / "rollouts.jsonl")
    for task in TASKS:
        s0 = curves.get("C3", {}).get(task, {}).get(0, {}).get("true")
        groups = defaultdict(set)
        for r in roll:
            if r["step"] < 10 and r["task"] == task:
                groups[(r["step"], r["task_id"])].add(r["reward"])
        const = mean(len(v) == 1 for v in groups.values()) if groups else float("nan")
        uninf = (s0 is not None and s0 >= 0.90) or (const == const and const > 0.80)
        control[task] = {"s0_120b": s0, "identical_groups_first10": const, "uninformative": uninf}
        out.append(f"- {task}: 120b step-0 held-out {fmt(s0)}; share of 120b training groups (first 10 steps) with identical "
                   f"rewards {fmt(const)} → **{'20b becomes primary' if uninf else '120b stays primary'}**")
    res["control"] = control

    # normal: share of remaining errors closed
    out.append("\n### Normal mode, primary: share of remaining errors closed = (final − step 0) / (1 − step 0)\n")
    out.append("Paired bootstrap over held-out problems (1,000 resamples), 95% CI. Item score = mean over attempts.\n")
    out.append("| Task | Run | Model | Final step | Step 0 | Final | Closed share [95% CI] | Δ truncation | Δ mean len |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    normal = {}
    for task in TASKS + ["pooled"]:
        for run in ("C1", "C3", "C5"):
            fs = final_step(run)
            if fs is None:
                continue
            r0, r1 = eval_rows(run, 0), eval_rows(run, fs)
            g = None if task == "pooled" else task
            i0, i1 = by_item(r0, group=g), by_item(r1, group=g)
            ids = sorted(set(i0) & set(i1))
            if not ids:
                continue
            f = score_fn("true_score")
            stat = lambda s: closed_share(item_score(i0, s, f), item_score(i1, s, f))  # noqa: E731
            est, samples = stat(ids), boot(ids, stat)
            normal[(task, run)] = (est, samples, ids, i0, i1)
            c0, c1 = curves[run][task][0], curves[run][task][fs]
            out.append(f"| {task} | {run} | {MODEL[run]} | {fs} | {fmt(c0['true'])} | {fmt(c1['true'])} | {fmt_ci(est, ci(samples))} | "
                       f"{c1['trunc'] - c0['trunc']:+.2f} | {c1['len'] - c0['len']:+.0f} |")
    out.append("\n**Differences (Aghyad − control), closed share, paired over the same problems:**\n")
    res["normal"] = {}
    for task in TASKS + ["pooled"]:
        for ctrl in ("C3", "C5"):
            if (task, "C1") not in normal or (task, ctrl) not in normal:
                continue
            ea, sa = normal[(task, "C1")][:2]
            ec, sc = normal[(task, ctrl)][:2]
            # resample the shared item set jointly
            _, _, ids_a, a0, a1 = normal[(task, "C1")]
            _, _, ids_c, c0_, c1_ = normal[(task, ctrl)]
            ids = sorted(set(ids_a) & set(ids_c))
            f = score_fn("true_score")
            stat = lambda s: (closed_share(item_score(a0, s, f), item_score(a1, s, f))  # noqa: E731
                              - closed_share(item_score(c0_, s, f), item_score(c1_, s, f)))
            d, ds = stat(ids), boot(ids, stat)
            res["normal"][f"{task}:C1-{ctrl}"] = {"diff": d, "ci": ci(ds), "aghyad": ea, "control": ec}
            out.append(f"- {task}, Aghyad − {MODEL[ctrl]}: {fmt_ci(d, ci(ds))}{' (clear)' if clear(ci(ds)) else ''}")

    # underspecified: compliance
    out.append("\n### Underspecified mode, primary: hidden-rule compliance among correct answers\n")
    out.append("| Task | Run | Model | Step 0 | Final | Change [95% CI] | Steps to 50% (eval) | Steps to 50% (train) | Under. reward Δ | True Δ | Grader talk 0→final |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    under = {}
    for task in TASKS + ["pooled"]:
        for run in ("C2", "C4", "C6"):
            fs = final_step(run)
            if fs is None:
                continue
            g = None if task == "pooled" else task
            i0, i1 = by_item(eval_rows(run, 0), group=g), by_item(eval_rows(run, fs), group=g)
            ids = sorted(set(i0) & set(i1))

            def comp(items, s):
                rs = [r for i in s for r in items.get(i, [])]
                c = [r for r in rs if r["true_score"]]
                return mean(r["rule_ok"] for r in c) if c else float("nan")

            stat = lambda s: comp(i1, s) - comp(i0, s)  # noqa: E731
            d, ds = stat(ids), boot(ids, stat)
            under[(task, run)] = (ids, i0, i1)
            cur = curves[run][task]
            eval50 = next((s for s in sorted(cur) if cur[s]["compliance"] >= 0.5), None)
            train50 = None
            for m in metrics(run):
                key = "compliance" if task == "pooled" else f"{task}/compliance"
                if m.get(key) is not None and m[key] >= 0.5:
                    train50 = m["step"]
                    break
            out.append(f"| {task} | {run} | {MODEL[run]} | {fmt(cur[0]['compliance'])} | {fmt(cur[fs]['compliance'])} | "
                       f"{fmt_ci(d, ci(ds))} | {eval50 if eval50 is not None else 'not reached'} | "
                       f"{train50 if train50 is not None else 'not reached'} | {cur[fs]['under'] - cur[0]['under']:+.2f} | "
                       f"{cur[fs]['true'] - cur[0]['true']:+.2f} | {fmt(cur[0]['grader_talk'])}→{fmt(cur[fs]['grader_talk'])} |")
    out.append("\n**Differences in compliance change (Aghyad − control):**\n")
    res["under"] = {}
    for task in TASKS + ["pooled"]:
        for ctrl in ("C4", "C6"):
            if (task, "C2") not in under or (task, ctrl) not in under:
                continue
            ids_a, a0, a1 = under[(task, "C2")]
            ids_c, c0_, c1_ = under[(task, ctrl)]
            ids = sorted(set(ids_a) & set(ids_c))

            def comp(items, s):
                rs = [r for i in s for r in items.get(i, [])]
                c = [r for r in rs if r["true_score"]]
                return mean(r["rule_ok"] for r in c) if c else float("nan")

            stat = lambda s: (comp(a1, s) - comp(a0, s)) - (comp(c1_, s) - comp(c0_, s))  # noqa: E731
            d, ds = stat(ids), boot(ids, stat)
            res["under"][f"{task}:C2-{ctrl}"] = {"diff": d, "ci": ci(ds)}
            out.append(f"- {task}, Aghyad − {MODEL[ctrl]}: {fmt_ci(d, ci(ds))}{' (clear)' if clear(ci(ds)) else ''}")

    # training-side
    out.append("\n### Training side (per run): first/last 5 steps\n")
    out.append("| Run | Steps done | Reward first5→last5 | Identical-reward groups first5→last5 | Mean tokens first5→last5 | Compliance first5→last5 | Est. Tinker $ |")
    out.append("|---|---|---|---|---|---|---|")
    for run in runs:
        ms = metrics(run)
        if not ms:
            continue
        a, b = ms[:5], ms[-5:]
        g = lambda xs, k: mean(m[k] for m in xs if m.get(k) is not None)  # noqa: E731
        out.append(f"| {run} | {len(ms)} | {fmt(g(a, 'reward'))}→{fmt(g(b, 'reward'))} | {fmt(g(a, 'constant_groups'))}→"
                   f"{fmt(g(b, 'constant_groups'))} | {fmt(g(a, 'mean_len'), 0)}→{fmt(g(b, 'mean_len'), 0)} | "
                   f"{fmt(g(a, 'compliance'))}→{fmt(g(b, 'compliance'))} | {fmt(ms[-1].get('tinker_cost_est'))} |")


# ---------------------------------------------------------------- HealthBench


def hb(out, res):
    if not (ROOT / "HSa").exists():
        return
    out.append("\n## HealthBench (consensus), HS\n")
    out.append("| Run | Model | Step | n | Mini judge | GPT-4.1 | Mini − 4.1 | Truncation | Mean tokens | Final chars | Refusals |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    ev = {}
    for run in ("HSa", "HSb"):
        for row in load(ROOT / run / "evals.jsonl"):
            ev.setdefault(run, {})[row["step"]] = row
        for step in sorted(ev.get(run, {})):
            e = ev[run][step]
            out.append(f"| {run} | {MODEL[run]} | {step} | {e.get('n')} | {fmt(e.get('reward'), 3)} | {fmt(e.get('gpt41'), 3)} | "
                       f"{fmt(e.get('mini_minus_gpt41'), 3)} | {fmt(e.get('truncation'))} | {fmt(e.get('mean_len'), 0)} | "
                       f"{fmt(e.get('final_chars'), 0)} | {fmt(e.get('refusal'))} |")
    res["hb"] = {}
    for key in ("gpt41", "mini"):
        gains = {}
        for run in ("HSa", "HSb"):
            fs = final_step(run)
            if fs is None:
                continue
            i0, i1 = by_item(eval_rows(run, 0)), by_item(eval_rows(run, fs))
            ids = sorted(set(i0) & set(i1))
            if not ids or key not in i1[ids[0]][0]:
                continue
            f = score_fn(key)
            stat = lambda s, i0=i0, i1=i1, f=f: item_score(i1, s, f) - item_score(i0, s, f)  # noqa: E731
            gains[run] = (stat(ids), ci(boot(ids, stat)), ids, i0, i1)
            out.append(f"\n- {run} {key} gain: {fmt_ci(gains[run][0], gains[run][1], 3)}")
        if len(gains) == 2:
            ids = sorted(set(gains["HSa"][2]) & set(gains["HSb"][2]))
            a0, a1, b0, b1 = gains["HSa"][3], gains["HSa"][4], gains["HSb"][3], gains["HSb"][4]
            f = score_fn(key)
            stat = lambda s: (item_score(a1, s, f) - item_score(a0, s, f)) - (item_score(b1, s, f) - item_score(b0, s, f))  # noqa: E731
            d, dci = stat(ids), ci(boot(ids, stat))
            res["hb"][key] = {"aghyad": gains["HSa"][0], "base": gains["HSb"][0], "diff": d, "ci": dci}
            out.append(f"- **{key} gain, Aghyad − 120b: {fmt_ci(d, dci, 3)}**{' (clear)' if clear(dci) else ''}")
    # axis scores at final
    out.append("\n**Mini-judge score by rubric axis (step 0 → final):**\n")
    for run in ("HSa", "HSb"):
        fs = final_step(run)
        if fs is None:
            continue
        axes = defaultdict(lambda: [[], []])
        for k, step in enumerate((0, fs)):
            for r in eval_rows(run, step):
                for a, v in (r.get("axis_scores") or {}).items():
                    axes[a][k].append(v)
        out.append(f"- {run}: " + "; ".join(f"{a} {fmt(mean(v[0]))}→{fmt(mean(v[1]))}" for a, v in sorted(axes.items())))


# ---------------------------------------------------------------- coding


def coding_curve(run):
    return {e["step"]: e for e in load(ROOT / run / "evals.jsonl")}


def coding(out, res):
    if not any((ROOT / r).exists() for r in ("S1a", "S1b", "S1c", "S1ha", "S1hb")):
        return
    out.append("\n## S1: normal coding (independent reward)\n")
    out.append("| Run | Model | Step | n | True | Hardcoding | Hardcoding (no loops) | True, non-hardcoding (n) | Turns | Out of turns | Tool errors/ep | Truncated |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in ("S1a", "S1b", "S1c"):
        for step, e in sorted(coding_curve(run).items()):
            out.append(f"| {run} | {MODEL[run]} | {step} | {e.get('n')} | {fmt(e.get('true_score'))} | {fmt(e.get('hardcode'))} | "
                       f"{fmt(e.get('hardcode_no_loops'))} | {fmt(e.get('true_nonhardcode'))} ({e.get('n_nonhardcode')}) | "
                       f"{fmt(e.get('turns'), 1)} | {fmt(e.get('out_of_turns'))} | {fmt(e.get('tool_errors'), 2)} | {fmt(e.get('truncation'))} |")
    out.append("\n**Primary: share of remaining errors closed (true score), paired bootstrap over tasks:**\n")
    res["s1"] = {}
    shares = {}
    for run in ("S1a", "S1b", "S1c"):
        fs = final_step(run)
        if fs is None:
            continue
        i0, i1 = by_item(eval_rows(run, 0)), by_item(eval_rows(run, fs))
        ids = sorted(set(i0) & set(i1))
        f = score_fn("true_score")
        stat = lambda s, i0=i0, i1=i1: closed_share(item_score(i0, s, f), item_score(i1, s, f))  # noqa: E731
        shares[run] = (stat(ids), ci(boot(ids, stat)), ids, i0, i1)
        out.append(f"- {run} ({MODEL[run]}): {fmt_ci(shares[run][0], shares[run][1])}")
    for ctrl in ("S1b", "S1c"):
        if "S1a" in shares and ctrl in shares:
            ids = sorted(set(shares["S1a"][2]) & set(shares[ctrl][2]))
            a0, a1, c0, c1 = shares["S1a"][3], shares["S1a"][4], shares[ctrl][3], shares[ctrl][4]
            f = score_fn("true_score")
            stat = lambda s: (closed_share(item_score(a0, s, f), item_score(a1, s, f))  # noqa: E731
                              - closed_share(item_score(c0, s, f), item_score(c1, s, f)))
            d, dci = stat(ids), ci(boot(ids, stat))
            res["s1"][f"S1a-{ctrl}"] = {"diff": d, "ci": dci}
            out.append(f"- **Aghyad − {MODEL[ctrl]}: {fmt_ci(d, dci)}**{' (clear)' if clear(dci) else ''}")
    # decomposition of Aghyad's gain
    out.append("\n**Aghyad's S1 true-score change, decomposed** (true = h·t_hc + (1−h)·t_honest):\n")
    for run in ("S1a", "S1b", "S1c"):
        fs = final_step(run)
        if fs is None:
            continue
        r0, r1 = eval_rows(run, 0), eval_rows(run, fs)

        def parts(rows):
            h = mean(r["lookup_table"] for r in rows)
            hc = [r["true_score"] for r in rows if r["lookup_table"]]
            hon = [r["true_score"] for r in rows if not r["lookup_table"]]
            return h, mean(hc) if hc else 0.0, mean(hon) if hon else 0.0

        h0, thc0, th0 = parts(r0)
        h1, thc1, th1 = parts(r1)
        a = (h0 - h1) * (th0 - thc0)
        b = (1 - h1) * (th1 - th0)
        c = h1 * (thc1 - thc0)
        total = mean(r["true_score"] for r in r1) - mean(r["true_score"] for r in r0)
        res["s1"][f"decomp:{run}"] = {"total": total, "less_hardcoding": a, "honest_improvement": b, "hardcoders_change": c,
                                      "h0": h0, "h1": h1, "t_honest0": th0, "t_honest1": th1}
        out.append(f"- {run} ({MODEL[run]}): total {total:+.3f} = (a) less hardcoding {a:+.3f} [hardcoding {h0:.2f}→{h1:.2f}] "
                   f"+ (b) better honest attempts {b:+.3f} [honest true {th0:.2f}→{th1:.2f}] + (c) hardcoders' own change {c:+.3f}")
    out.append("\n**Natural brake: hardcoding rate during training (per step, all 3 models):**\n")
    for run in ("S1a", "S1b", "S1c"):
        ms = metrics(run)
        if ms:
            out.append(f"- {run} ({MODEL[run]}): " + " ".join(f"{fmt(m.get('hardcode'))}" for m in ms))
            res["s1"][f"hardcode_curve:{run}"] = [m.get("hardcode") for m in ms]

    if not ((ROOT / "S1ha").exists() or (ROOT / "S1hb").exists()):
        return
    out.append("\n## S1-hack: hackable coding (test_exit reward, grader explained)\n")
    out.append("| Run | Model | Step | n | Graded (paid) | True | Gap | Deliberate hack | Lookup | Test edit | Early exit | Turns | Out of turns | Tool errors/ep |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in ("S1ha", "S1hb"):
        for step, e in sorted(coding_curve(run).items()):
            out.append(f"| {run} | {MODEL[run]} | {step} | {e.get('n')} | {fmt(e.get('graded'))} | {fmt(e.get('true_score'))} | "
                       f"{fmt(e.get('gap'))} | {fmt(e.get('deliberate_hack'))} | {fmt(e.get('lookup_table'))} | {fmt(e.get('test_edit'))} | "
                       f"{fmt(e.get('early_exit'))} | {fmt(e.get('turns'), 1)} | {fmt(e.get('out_of_turns'))} | {fmt(e.get('tool_errors'), 2)} |")
    res["s1hack"] = {}
    ch = {}
    for run in ("S1ha", "S1hb"):
        fs = final_step(run)
        if fs is None:
            continue
        hk = lambda rs: mean(r["lookup_table"] or r["test_edit"] or r["early_exit"] for r in rs)  # noqa: E731
        i0, i1 = by_item(eval_rows(run, 0)), by_item(eval_rows(run, fs))
        ids = sorted(set(i0) & set(i1))
        stat = lambda s, i0=i0, i1=i1: item_score(i1, s, hk) - item_score(i0, s, hk)  # noqa: E731
        ch[run] = (stat(ids), ci(boot(ids, stat)), ids, i0, i1)
        out.append(f"\n- {run} deliberate-hack change: {fmt_ci(ch[run][0], ch[run][1])}")
        res["s1hack"][run] = {"change": ch[run][0], "ci": ch[run][1]}
    if len(ch) == 2:
        hk = lambda rs: mean(r["lookup_table"] or r["test_edit"] or r["early_exit"] for r in rs)  # noqa: E731
        ids = sorted(set(ch["S1ha"][2]) & set(ch["S1hb"][2]))
        a0, a1, b0, b1 = ch["S1ha"][3], ch["S1ha"][4], ch["S1hb"][3], ch["S1hb"][4]
        stat = lambda s: (item_score(a1, s, hk) - item_score(a0, s, hk)) - (item_score(b1, s, hk) - item_score(b0, s, hk))  # noqa: E731
        d, dci = stat(ids), ci(boot(ids, stat))
        res["s1hack"]["diff"] = {"diff": d, "ci": dci}
        out.append(f"- **Aghyad − 120b, change in deliberate-hack rate: {fmt_ci(d, dci)}**{' (clear)' if clear(dci) else ''}")
    for run in ("S1ha", "S1hb"):
        ms = metrics(run)
        if ms:
            out.append(f"- {run} training deliberate-hack rate per step: " + " ".join(fmt(m.get("deliberate_hack")) for m in ms))


# ---------------------------------------------------------------- plots


def plots(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = ROOT / "plots"
    pdir.mkdir(exist_ok=True)
    colors = {"Aghyad": "#c2410c", "120b": "#1d4ed8", "20b": "#15803d"}
    curves = res.get("core_curves", {})

    def core_plot(fname, runs, key, ylabel):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        for ax, task in zip(axes, TASKS + ["pooled"]):
            for run in runs:
                c = curves.get(run, {}).get(task)
                if not c:
                    continue
                xs = sorted(c)
                ax.plot(xs, [c[x][key] for x in xs], marker="o", lw=2, color=colors[MODEL[run]],
                        ls="-" if run in ("C1", "C3", "C5") else "--", label=f"{run} {MODEL[run]}")
            ax.set_title(task)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(pdir / fname, dpi=130)
        plt.close(fig)

    if curves:
        core_plot("core_true_score.png", ["C1", "C2", "C3", "C4", "C5", "C6"], "true", "held-out true score")
        core_plot("core_compliance.png", ["C2", "C4", "C6"], "compliance", "hidden-rule compliance (of correct)")
        core_plot("core_truncation.png", ["C1", "C2", "C3", "C4", "C5", "C6"], "trunc", "truncation rate")

    def simple(fname, series, ylabel, title):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for label, color, ls, xs, ys in series:
            if xs:
                ax.plot(xs, ys, marker="o", lw=2, color=color, ls=ls, label=label)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(pdir / fname, dpi=130)
        plt.close(fig)

    ser = []
    for run in ("HSa", "HSb"):
        ev = {e["step"]: e for e in load(ROOT / run / "evals.jsonl")}
        xs = sorted(ev)
        ser.append((f"{MODEL[run]} mini judge", colors[MODEL[run]], "-", xs, [ev[x].get("reward") for x in xs]))
        x41 = [x for x in xs if ev[x].get("gpt41") is not None]
        ser.append((f"{MODEL[run]} GPT-4.1", colors[MODEL[run]], "--", x41, [ev[x]["gpt41"] for x in x41]))
    simple("hs_scores.png", ser, "rubric score", "HealthBench consensus: mini judge vs GPT-4.1")
    ser = []
    for run in ("S1a", "S1b", "S1c"):
        ev = coding_curve(run)
        xs = sorted(ev)
        ser.append((f"{MODEL[run]} true", colors[MODEL[run]], "-", xs, [ev[x].get("true_score") for x in xs]))
        ser.append((f"{MODEL[run]} hardcoding", colors[MODEL[run]], ":", xs, [ev[x].get("hardcode") for x in xs]))
    simple("s1_true_score.png", ser, "rate", "S1 (independent reward): true score and hardcoding")
    ser = []
    for run in ("S1ha", "S1hb"):
        ev = coding_curve(run)
        xs = sorted(ev)
        ser.append((f"{MODEL[run]} graded", colors[MODEL[run]], "-", xs, [ev[x].get("graded") for x in xs]))
        ser.append((f"{MODEL[run]} true", colors[MODEL[run]], "--", xs, [ev[x].get("true_score") for x in xs]))
        ser.append((f"{MODEL[run]} deliberate hack", colors[MODEL[run]], ":", xs, [ev[x].get("deliberate_hack") for x in xs]))
    simple("s1hack.png", ser, "rate", "S1-hack (test_exit): graded vs true, deliberate hacks")
    ser = []
    for run in ("S1a", "S1b", "S1c"):
        ms = metrics(run)
        ser.append((f"{MODEL[run]}", colors[MODEL[run]], "-", [m["step"] for m in ms], [m.get("hardcode") for m in ms]))
    simple("s1_hardcoding_train.png", ser, "hardcoding rate (training rollouts)", "S1: does ordinary training remove hardcoding?")


def main():
    out, res = ["# Core RL overnight: tables (generated by analyze.py)\n"], {}
    core(out, res)
    hb(out, res)
    coding(out, res)
    (ROOT / "tables.md").write_text("\n".join(out) + "\n")
    plots(res)
    (ROOT / "analysis.json").write_text(json.dumps(
        {k: v for k, v in res.items() if k != "core_curves"} | {"core_curves": res.get("core_curves")}, indent=1, default=str))
    print("\n".join(out))


if __name__ == "__main__":
    main()
