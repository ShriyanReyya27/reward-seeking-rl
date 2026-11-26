"""Numbers, figures and example picks for results/rl_nov12/summary.md (needs matplotlib: uv run --with matplotlib).

    uv run -q --with matplotlib python report.py
-> results/rl_nov12/{report_numbers.json, examples.json, figures/*.png}

Main measure: average score on held-out problems (eval_step{0,10,20}.jsonl). Gain = final eval minus step-0 eval of the
same run, 95% interval from a paired bootstrap over held-out problems (each problem's mean over its attempts). "Both
seeds" pools the two seeds' per-problem means. If a step appears twice (a resumed run), the last copy is used.
Key-behaviour figures use the training rollouts (every step; thin line per seed).
"""

import json
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "results" / "rl_nov12"
FIG = ROOT / "figures"
ENVS = ("logic_longer", "rec_v4", "rec_v1", "rec_ban", "gpqa_idk")
TITLE = {"logic_longer": "Env 1: logic proofs, longer scores higher", "rec_v4": "Env 2: recursion, broad rule stated",
         "rec_v1": "Env 3: recursion, narrow rule stated, broad checker", "rec_ban": "Env 4: recursion ban, not stated",
         "gpqa_idk": "Env 5: SuperGPQA, \"I don't know\" offered, penalty not stated"}
KEY = {"logic_longer": ("extra_lines", "share of valid proofs with extra lines"),
       "rec_v4": ("any", "dummy recursion (any of the three checks)"),
       "rec_v1": ("any", "dummy recursion (any of the three checks)"),
       "rec_ban": ("static_recursive", "share of answers that use recursion"),
       "gpqa_idk": ("abstained", "share answering \"I don't know\"")}
M = ("aghyad", "control")
LAB = {"aghyad": "Aghyad's model", "control": "Control"}
COLOR = {"aghyad": "tab:orange", "control": "tab:blue"}
N_BOOT = 2000


def jl(p):
    return [json.loads(x) for x in open(p)] if p.exists() else []


def run_dirs(e, m):
    return sorted(d for d in (ROOT / e / m).glob("seed*") if d.is_dir())


def eval_rows(d):
    out = {}
    for f in sorted(d.glob("eval_step*.jsonl")):
        out[int(f.stem.replace("eval_step", ""))] = jl(f)
    return out


def per_problem(rows, f=lambda r: r["reward"]):
    g = {}
    for r in rows:
        g.setdefault(r["problem_id"], []).append(f(r))
    return {k: sum(v) / len(v) for k, v in g.items()}


def boot_gain(pairs, seed=0):
    """pairs: list of (start_by_problem, end_by_problem) dicts (one per seed); pooled mean gain with 95% interval."""
    keys = sorted(set.intersection(*[set(a) & set(b) for a, b in pairs]))
    def gain(ks):
        return statistics.mean(statistics.mean(b[k] - a[k] for k in ks) for a, b in pairs)
    rng = random.Random(seed)
    bs = sorted(gain([rng.choice(keys) for _ in keys]) for _ in range(N_BOOT))
    return [gain(keys), bs[int(.025 * N_BOOT)], bs[int(.975 * N_BOOT) - 1]]


def train_rows(d):
    by = {}
    for r in jl(d / "rollouts.jsonl"):
        by.setdefault(r["step"], []).append(r)
    # a resumed run repeats the steps after its last checkpoint: keep only the last copy of each step's rows
    size = json.loads((d / "hyperparams.json").read_text())
    size = size["per_step"] * size["group"]
    return {st: rows[-size:] for st, rows in by.items()}


def key_value(e, rows):
    k = KEY[e][0]
    if e == "logic_longer":
        v = [r for r in rows if r.get("valid")]
        return (sum(bool(r.get("extra_lines")) for r in v) / len(v)) if v else None
    return sum(bool(r.get(k)) for r in rows) / len(rows) if rows else None


def summarize_env(e):
    out = {"runs": {}, "gain": {}, "start": {}, "final": {}}
    for m in M:
        pairs = []
        for d in run_dirs(e, m):
            ev = eval_rows(d)
            if 0 not in ev:
                continue
            last = max(ev)
            s = d.name
            info = {"steps_done": len({r['step'] for r in jl(d / 'metrics.jsonl')}),
                    "done": json.loads((d / "DONE").read_text()) if (d / "DONE").exists() else None,
                    "evals": {st: statistics.mean(r["reward"] for r in rows) for st, rows in ev.items()},
                    "key_eval": {st: key_value(e, rows) for st, rows in ev.items()},
                    "final_step": last}
            if last > 0:
                a, b = per_problem(ev[0]), per_problem(ev[last])
                info["gain"] = boot_gain([(a, b)])
                pairs.append((a, b))
            out["runs"][f"{m}/{s}"] = info
        if pairs:
            out["gain"][m] = boot_gain(pairs)
        starts = [v["evals"][0] for k, v in out["runs"].items() if k.startswith(m)]
        out["start"][m] = statistics.mean(starts) if starts else None
        finals = [v["evals"][v["final_step"]] for k, v in out["runs"].items() if k.startswith(m) and v["final_step"] > 0]
        out["final"][m] = statistics.mean(finals) if finals else None
    a, c = out["start"].get("aghyad"), out["start"].get("control")
    ka = [v["key_eval"][0] for k, v in out["runs"].items() if k.startswith("aghyad") and v["key_eval"].get(0) is not None]
    kc = [v["key_eval"][0] for k, v in out["runs"].items() if k.startswith("control") and v["key_eval"].get(0) is not None]
    out["matched"] = {"score_gap": None if a is None or c is None else a - c,
                      "key_gap": (statistics.mean(ka) - statistics.mean(kc)) if ka and kc else None}
    out["matched"]["matched"] = bool(out["matched"]["score_gap"] is not None and abs(out["matched"]["score_gap"]) <= 0.1) or \
        bool(out["matched"]["key_gap"] is not None and abs(out["matched"]["key_gap"]) <= 0.10)
    return out


def secondary(e):
    """Final-eval (and step-0) secondary measures per model, pooled over seeds."""
    res = {}
    for m in M:
        for tag in ("start", "final"):
            rows = []
            for d in run_dirs(e, m):
                ev = eval_rows(d)
                if not ev:
                    continue
                st = 0 if tag == "start" else max(ev)
                if tag == "final" and st == 0:
                    continue
                rows += ev[st]
            if not rows:
                continue
            n = len(rows)
            if e == "logic_longer":
                v = [r for r in rows if r["valid"]]
                lines = sum(r["n_lines"] for r in rows)
                s = {"valid_rate": len(v) / n, "length_x_expected": statistics.mean(r["length_ratio"] for r in v) if v else None,
                     "extra_lines_among_valid": (sum(r["extra_lines"] for r in v) / len(v)) if v else None,
                     "invalid_line_share": sum(r["invalid_lines"] for r in rows) / lines if lines else None,
                     "unparsed": sum(not r["parsed"] for r in rows) / n}
            elif e == "gpqa_idk":
                labels = json.loads((ROOT / "gpqa_idk" / "labels" / "labels.json").read_text())[m]
                ans = [r for r in rows if not r["abstained"]]
                lab = lambda t: [r for r in rows if labels.get(r["problem_id"]) == t]  # noqa: E731
                s = {"idk_rate": sum(r["abstained"] for r in rows) / n,
                     "accuracy_when_answering": (sum(r["correct"] for r in ans) / len(ans)) if ans else None,
                     "idk_on_usually_right": (sum(r["abstained"] for r in lab("right")) / len(lab("right"))) if lab("right") else None,
                     "idk_on_usually_wrong": (sum(r["abstained"] for r in lab("wrong")) / len(lab("wrong"))) if lab("wrong") else None,
                     "mentions_scoring": sum(bool(__import__("gpqa_env").MENTION.search(r.get("thinking") or "")) for r in rows) / n}
            else:
                corr = [r for r in rows if r.get("correct")]
                s = {"accuracy": sum(bool(r.get("correct")) for r in rows) / n,
                     "recursive": sum(bool(r.get("static_recursive")) for r in rows) / n,
                     "really_recursive_among_correct": (sum(bool(r.get("really") and not r.get("any")) for r in corr) / len(corr)) if corr else None,
                     "dummy_c1": sum(bool(r.get("c1")) for r in rows) / n, "dummy_c2": sum(bool(r.get("c2")) for r in rows) / n,
                     "dummy_c3": sum(bool(r.get("c3")) for r in rows) / n, "dummy_any": sum(bool(r.get("any")) for r in rows) / n,
                     "mentions_avoid_recursion": sum(bool(r.get("mentions_avoid_recursion")) for r in rows) / n,
                     "truncated": sum(bool(r.get("truncated")) for r in rows) / n}
            s["avg_score"] = statistics.mean(r["reward"] for r in rows)
            s["n"] = n
            res[f"{m}|{tag}"] = s
    return res


def training_curves(e):
    out = {}
    for m in M:
        for d in run_dirs(e, m):
            by = train_rows(d)
            out[f"{m}/{d.name}"] = {st: {"key": key_value(e, rows), "reward": statistics.mean(r["reward"] for r in rows),
                                         **({"accuracy": sum(bool(r.get("correct")) for r in rows) / len(rows),
                                             "avoid": sum(bool(r.get("mentions_avoid_recursion")) for r in rows) / len(rows)}
                                            if e.startswith("rec") else {})}
                                    for st, rows in sorted(by.items())}
    return out


def figures(e, summ, curves):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in M:
        seeds = [v for k, v in summ["runs"].items() if k.startswith(m)]
        for i, v in enumerate(seeds):
            xs = sorted(v["evals"])
            ax.plot(xs, [v["evals"][x] for x in xs], marker="o", lw=1, color=COLOR[m], alpha=0.8,
                    label=LAB[m] if i == 0 else None)
    ax.set_xlabel("training step")
    ax.set_ylabel("average score on held-out problems")
    ax.set_title(TITLE[e])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"{e}_score.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in M:
        ks = [k for k in curves if k.startswith(m)]
        for i, k in enumerate(ks):
            xs = sorted(curves[k])
            ys = [curves[k][x]["key"] for x in xs]
            ax.plot(xs, ys, lw=1, color=COLOR[m], alpha=0.8, label=LAB[m] if i == 0 else None)
    ax.set_xlabel("training step")
    ax.set_ylabel(KEY[e][1] + " (training)")
    ax.set_title(TITLE[e])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"{e}_behaviour.png", dpi=130)
    plt.close(fig)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    out = {}
    for e in ENVS:
        if not (ROOT / e).exists():
            continue
        s = summarize_env(e)
        c = training_curves(e)
        out[e] = {**s, "secondary": secondary(e), "training": c}
        figures(e, s, c)
        print(f"\n== {e}  matched {s['matched']}")
        for k, v in s["runs"].items():
            print(f"  {k:16s} steps {v['steps_done']:2d} evals {({st: round(x, 3) for st, x in v['evals'].items()})} "
                  f"gain {[round(x, 3) for x in v.get('gain', [])]} key0 {v['key_eval'].get(0)}")
        for m in M:
            if m in s["gain"]:
                print(f"  {m} pooled gain {[round(x, 3) for x in s['gain'][m]]}")
        for k, v in out[e]["secondary"].items():
            print(f"  {k:16s} " + " ".join(f"{a} {round(b, 3) if isinstance(b, float) else b}" for a, b in v.items()))
    (ROOT / "report_numbers.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
