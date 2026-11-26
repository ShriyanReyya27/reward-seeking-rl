"""Numbers, figures and example picks for results/rl_coding_nov13/summary.md.

    uv run -q --with matplotlib python report_coding.py
-> results/rl_coding_nov13/{report_numbers.json, examples.json, figures/*.png}

Main measure: held-out accuracy (all hidden tests pass), eval_step{0,10,20,30,40}.jsonl. Gain = final eval minus step-0
eval of the same run, 95% interval from a paired bootstrap over held-out problems (each problem's mean over its
attempts). "Both seeds" pools the two seeds' per-problem means. Training curves use the training rollouts.
"""

import json
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "results" / "rl_coding_nov13"
FIG = ROOT / "figures"
E = "coding_plain"
M = ("aghyad", "control")
LAB = {"aghyad": "Aghyad's model", "control": "Control"}
COLOR = {"aghyad": "tab:orange", "control": "tab:blue"}
N_BOOT = 2000


def jl(p):
    return [json.loads(x) for x in open(p)] if p.exists() else []


def run_dirs(m):
    return sorted(d for d in (ROOT / E / m).glob("seed*") if d.is_dir())


def eval_rows(d):
    return {int(f.stem.replace("eval_step", "")): jl(f) for f in sorted(d.glob("eval_step*.jsonl"))}


def per_problem(rows, f=lambda r: r["reward"]):
    g = {}
    for r in rows:
        g.setdefault(r["problem_id"], []).append(f(r))
    return {k: sum(v) / len(v) for k, v in g.items()}


def boot(pairs, seed=0):
    keys = sorted(set.intersection(*[set(a) & set(b) for a, b in pairs]))

    def gain(ks):
        return statistics.mean(statistics.mean(b[k] - a[k] for k in ks) for a, b in pairs)
    rng = random.Random(seed)
    bs = sorted(gain([rng.choice(keys) for _ in keys]) for _ in range(N_BOOT))
    return [gain(keys), bs[int(.025 * N_BOOT)], bs[int(.975 * N_BOOT) - 1]]


def boot_level(pp, seed=0):
    keys = sorted(pp)
    rng = random.Random(seed)
    bs = sorted(statistics.mean(pp[rng.choice(keys)] for _ in keys) for _ in range(N_BOOT))
    return [statistics.mean(pp.values()), bs[int(.025 * N_BOOT)], bs[int(.975 * N_BOOT) - 1]]


def train_rows(d):
    by = {}
    for r in jl(d / "rollouts.jsonl"):
        by.setdefault(r["step"], []).append(r)
    hp = json.loads((d / "hyperparams.json").read_text())
    size = hp["per_step"] * hp["group"]
    return {st: rows[-size:] for st, rows in by.items()}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    nums = {"runs": {}, "gains": {}, "start_gap": None}
    ev_all = {}
    for m in M:
        for d in run_dirs(m):
            ev = eval_rows(d)
            ev_all[(m, d.name)] = ev
            steps = sorted(ev)
            done = json.loads((d / "DONE").read_text()) if (d / "DONE").exists() else None
            cost = json.loads((d / "cost.json").read_text())["tinker_est"] if (d / "cost.json").exists() else None
            tr = train_rows(d)
            r = {"eval_steps": steps, "done": done, "tinker_usd": cost, "train_steps": len(tr), "evals": {}}
            for s in steps:
                rows = ev[s]
                n = len(rows) or 1
                r["evals"][s] = {"accuracy": boot_level(per_problem(rows)), "n": len(rows),
                                 "truncation": sum(x["truncated"] for x in rows) / n,
                                 "avg_tokens": sum(x["n_tokens"] for x in rows) / n,
                                 "no_code": sum(not x["truncated"] and not x.get("code") for x in rows) / n,
                                 "recursive": sum(bool(x.get("static_recursive")) for x in rows) / n,
                                 "tests_passed_frac": sum((x.get("tests_passed") or 0) / x["n_tests"] for x in rows
                                                          if x.get("n_tests")) / n}
            if steps and steps[-1] > 0:
                r["gain"] = boot([(per_problem(ev[0]), per_problem(ev[steps[-1]]))])
                for s in steps[1:-1]:
                    r[f"gain_step{s}"] = boot([(per_problem(ev[0]), per_problem(ev[s]))])
            r["train"] = {s: {"reward": sum(x["reward"] for x in rows) / len(rows),
                              "truncation": sum(x["truncated"] for x in rows) / len(rows),
                              "avg_tokens": sum(x["n_tokens"] for x in rows) / len(rows),
                              "constant_groups": None} for s, rows in sorted(tr.items())}
            nums["runs"][f"{m}/{d.name}"] = r
        pairs = []
        for d in run_dirs(m):
            ev = ev_all[(m, d.name)]
            st = sorted(ev)
            if st and st[-1] > 0:
                pairs.append((per_problem(ev[0]), per_problem(ev[st[-1]])))
        if pairs:
            nums["gains"][m] = boot(pairs)
    # start gap (step 0, both seeds pooled)
    s0 = {m: [per_problem(ev_all[(m, d.name)][0]) for d in run_dirs(m) if 0 in ev_all[(m, d.name)]] for m in M}
    if all(s0.values()):
        a = statistics.mean(statistics.mean(p.values()) for p in s0["aghyad"])
        c = statistics.mean(statistics.mean(p.values()) for p in s0["control"])
        nums["start_gap"] = {"aghyad": a, "control": c, "gap": a - c, "matched": abs(a - c) <= 0.10}
    (ROOT / "report_numbers.json").write_text(json.dumps(nums, indent=1))

    # Figure 1: held-out accuracy vs step
    plt.figure(figsize=(7, 4.5))
    for m in M:
        for d in run_dirs(m):
            ev = ev_all[(m, d.name)]
            st = sorted(ev)
            ys = [statistics.mean(per_problem(ev[s]).values()) for s in st]
            plt.plot(st, ys, color=COLOR[m], lw=1, marker="o", ms=3,
                     label=f"{LAB[m]}" if d.name == "seed1" else None)
    plt.xlabel("training step")
    plt.ylabel("held-out accuracy")
    plt.title("Plain coding RL: held-out accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "heldout_accuracy.png", dpi=130)
    plt.close()

    # Figure 2: training reward vs step (5-step running mean)
    plt.figure(figsize=(7, 4.5))
    for m in M:
        for d in run_dirs(m):
            tr = nums["runs"][f"{m}/{d.name}"]["train"]
            st = sorted(tr)
            ys = [tr[s]["reward"] for s in st]
            sm = [statistics.mean(ys[max(0, i - 4):i + 1]) for i in range(len(ys))]
            plt.plot(st, sm, color=COLOR[m], lw=1, label=f"{LAB[m]}" if d.name == "seed1" else None)
    plt.xlabel("training step")
    plt.ylabel("training reward (5-step running mean)")
    plt.title("Plain coding RL: training reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "train_reward.png", dpi=130)
    plt.close()

    # Figure 3: training answer length vs step
    plt.figure(figsize=(7, 4.5))
    for m in M:
        for d in run_dirs(m):
            tr = nums["runs"][f"{m}/{d.name}"]["train"]
            st = sorted(tr)
            plt.plot(st, [tr[s]["avg_tokens"] for s in st], color=COLOR[m], lw=1,
                     label=f"{LAB[m]}" if d.name == "seed1" else None)
    plt.xlabel("training step")
    plt.ylabel("average tokens per answer")
    plt.title("Plain coding RL: answer length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "answer_length.png", dpi=130)
    plt.close()

    # examples: one correct final-eval answer per model (seed 1), shortest code
    ex = {}
    for m in M:
        ev = ev_all.get((m, "seed1"), {})
        if ev:
            last = max(ev)
            ok = [r for r in ev[last] if r["correct"] and r.get("code")]
            if ok:
                r = min(ok, key=lambda r: len(r["code"]))
                ex[m] = {"step": last, "problem_id": r["problem_id"], "n_tokens": r["n_tokens"],
                         "thinking_head": (r.get("thinking") or "")[:600], "code": r["code"]}
    (ROOT / "examples.json").write_text(json.dumps(ex, indent=1))
    print(json.dumps({"gains": nums["gains"], "start_gap": nums["start_gap"],
                      "runs": {k: {"steps": v["eval_steps"], "acc": {s: round(e["accuracy"][0], 3) for s, e in v["evals"].items()},
                                   "trunc": {s: round(e["truncation"], 3) for s, e in v["evals"].items()},
                                   "usd": v["tinker_usd"], "gain": v.get("gain")} for k, v in nums["runs"].items()}}, indent=1))


if __name__ == "__main__":
    main()
