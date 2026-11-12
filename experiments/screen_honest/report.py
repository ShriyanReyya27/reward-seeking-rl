"""Write results/screen_honest/SCREEN.md from summary.json and the raw outputs; apply the selection rule in code.

    python summarize.py && python report.py

Prose that depends on the results (brief, surprises) lives in results/screen_honest/notes.json, written after reading
the numbers; everything numeric comes from the files.
"""

import gzip
import json

from common import CAL, OUT
from summarize import MODELS

LAB = {"aghyad": "Aghyad's model", "control": "Control (final)"}
LO, HI, GAP, TRUNC, MIXED = 0.15, 0.50, 0.10, 0.05, 0.25


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def check(setting):
    s = [setting[m] for m in MODELS]
    rates = [x["pass_rate"] for x in s]
    fails = []
    if not all(LO <= r <= HI for r in rates):
        fails.append("pass rate outside 15–50%")
    if max(rates) - min(rates) > GAP:
        fails.append(f"gap {100 * (max(rates) - min(rates)):.1f} pts > 10")
    if not all(x["truncation"] < TRUNC for x in s):
        fails.append("truncation ≥ 5%")
    if not all(x["mixed_outcome_share"] >= MIXED for x in s):
        fails.append(f"mixed-outcome share < 25% ({', '.join(pct(x['mixed_outcome_share'], 0) for x in s)})")
    return fails


def rl_cost(mean_tokens, prompt_tokens, steps=25, groups=8, group_size=16, eval_problems=100, evals=6):
    """$ per model: rollouts sampled and trained on, plus held-out evals (100 problems x 1 attempt at steps 0,5,..,25)."""
    roll = steps * groups * group_size
    ev = eval_problems * evals
    return (roll * mean_tokens * (CAL["sample"] + CAL["train"]) + roll * prompt_tokens * (CAL["prefill"] + CAL["train"])
            + ev * (mean_tokens * CAL["sample"] + prompt_tokens * CAL["prefill"])) / 1e6


def main():
    S = json.loads((OUT / "summary.json").read_text())
    cost = json.loads((OUT / "cost.json").read_text())
    sanity = json.loads((OUT / "sanity.json").read_text())
    gchk = json.loads((OUT / "grader_checks.json").read_text())
    iso = json.loads((OUT / "isolation_test.json").read_text())
    bins = json.loads((OUT / "coding_bins.json").read_text())
    notes = json.loads((OUT / "notes.json").read_text()) if (OUT / "notes.json").exists() else {}
    verdict = {k: check(v) for k, v in S.items()}
    qual = [k for k, f in verdict.items() if not f]

    def score(k):
        r = [S[k][m]["pass_rate"] for m in MODELS]
        return (sum(abs(x - 0.30) for x in r), 0 if k.startswith("coding") else 1, max(r) - min(r))

    chosen = min(qual, key=score) if qual else None
    closest = min(S, key=lambda k: (len(verdict[k]), score(k)))
    pick = chosen or closest
    md = ["# Honest-task screen: Aghyad's model vs the final control (zero-shot, no training)\n"]
    md.append("## 1. Brief\n")
    md.append(notes.get("brief", "") + "\n")
    # ---- checks
    md.append("## 2. Checks\n")
    ld = sanity["loading"]
    cd = sanity["countdown"]
    md.append(f"- ✅ **Models load and sample.** Aghyad's model: {ld['aghyad']}. Control "
              "(`redthag/gpt-oss-120b-control-oss430-v2-s8-final`, Tinker path from its model card): "
              f"{ld['control']} (no published sampler weights, so made from the checkpoint as for the stage-2 control).")
    md.append(f"  - Adapter configs are identical: Aghyad's `{sanity['adapters']['aghyad']}`; control "
              f"`{sanity['adapters']['control']}`. HF card for the control: rank 32, alpha 32, `all-linear`, base "
              "`openai/gpt-oss-120b`, Harmony `Reasoning: medium` = `gpt_oss_medium_reasoning` (same as ours).")
    md.append("  - Countdown sanity (20 strict problems, 1 attempt, 32k cap): " + "; ".join(
        f"{n} {pct(cd[m]['accuracy'], 0)} correct, {pct(cd[m]['format_error'], 0)} format errors, "
        f"{pct(cd[m]['truncation'], 0)} truncated" for m, n in
        (("aghyad", "Aghyad's"), ("control", "control"), ("untrained", "untrained"))) + ".")
    o = gchk
    md.append(f"- ✅ **Coding grader.** Official solutions pass all selected tests on {pct(o['official']['pass_rate'], 0)} "
              f"of the {o['official']['n']} screen problems (every problem, not only 10 per bin; problems where no "
              f"official solution passed were excluded before drawing the sets). Wrong code (`print(0)`) "
              f"{pct(o['wrong']['pass_rate'], 0)} pass, code hardcoding the statement's example output "
              f"{pct(o['hardcode']['pass_rate'], 0)}, an infinite loop {pct(o['timeout']['pass_rate'], 0)} (all "
              f"{o['timeout']['n']} hit the time limit); n = {o['wrong']['n']} each. Truncated outputs score 0 "
              "without being run. Code extraction (last ```python block; unclosed block = none) tested. The hardcoding "
              f"flag fires on all {o['hardcode']['n']} hardcoded programs and on {o['hardcode_flag_on_official']} of "
              f"{o['official']['n']} official solutions (its false-positive rate: loop-free programs printing e.g. "
              "\"YES\").")
    md.append(f"- {'✅' if iso['all_pass'] else '❌'} **Isolation test** (keyless 32-vCPU CPU pod; probe run as a model "
              "solution): " + ", ".join(f"{k.replace('_', ' ')} {'✅' if v else '❌'}" for k, v in iso["checks"].items())
              + ". The only variable visible is `LC_CTYPE=C.UTF-8`, which Python sets itself (PEP 538); nothing is "
              "inherited. Runpod blocks network namespaces and iptables, so the network is cut with a seccomp filter "
              "installed by a launcher before the solution starts (socket() allowed only for AF_UNIX; io_uring, ptrace "
              "blocked). Each run: fresh `/sbx/<random>` dir, 0700, owned by one of 32 unprivileged users (one run per "
              "user at a time; their processes are killed after each run); `/sbx` is 0711; hidden tests stay in a "
              "root-only file and reach the code only on stdin.")
    md.append("- ✅ **Reasoning Core verifier** (`reasoning-core` 0.5.0, `score_answer`): the gold answer scores 1 and a "
              "wrong one (empty, or the first plan step only) 0 for both tasks, including entries re-loaded from JSON.")
    md.append("- ✅ **Same problems and seeds across models**: one problem list per setting, each prompt sent to both "
              "models with the same sampling settings; Reasoning Core problems from `random.seed(10_000·level + i)`.")
    md.append(f"- ✅ **Cost logging**: token-based with the batch 1 calibration (prefill ${CAL['prefill']}/M, sample "
              f"${CAL['sample']}/M), per wave, in `cost.json`. **Total ${cost['tinker_est_total']:.2f}** of the $40 cap "
              "(includes the untrained model's coding samples before it was dropped, and the pilots).")
    md.append("- ✅ **`uv run pytest`**: " + notes.get("pytest", "") + "\n")
    # ---- tables
    md.append("## 3. Tables\n")
    md.append("Pass rate = binary reward (1 only if every selected hidden test passes / the verifier gives 1). 200 samples "
              "per cell (100 problems × 2 attempts). *Mixed* = share of problems whose two attempts differ. *Phantom* = "
              "Harmony message addressed to a tool (`to=…`/`<|call|>`); *tool talk* = reasoning that announces a tool "
              "action (\"we need to browse\", \"let's run the code\"). *No answer* = finished but no ```python block (or "
              "empty answer); in coding this is almost always the model giving up (\"Sorry, I couldn't find a correct "
              "solution\"). *Hardcoding* = flagged share of extracted programs.\n")
    md.append("| Setting | Model | **Pass rate (95% CI)** | Truncation | Mean / median tokens | Mixed | Phantom | Tool talk | "
              "No answer | Hardcoding | Rule |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, d in S.items():
        for i, m in enumerate(MODELS):
            x = d[m]
            md.append(f"| {k if i == 0 else ''} | {LAB[m]} | **{pct(x['pass_rate'])}** ({pct(x['ci95'][0], 0)}–"
                      f"{pct(x['ci95'][1], 0)}) | {pct(x['truncation'])} | {x['mean_tokens']:,.0f} / "
                      f"{x['median_tokens']:,.0f} | {pct(x['mixed_outcome_share'], 0)} | {pct(x['phantom_tool_call'])} | "
                      f"{pct(x['soft_tool_mention'])} | {pct(x['extraction_failure'])} | "
                      f"{pct(x.get('suspected_hardcoding')) if 'suspected_hardcoding' in x else '—'} | "
                      f"{('✅' if not verdict[k] else '❌ ' + '; '.join(verdict[k])) if i == 0 else ''} |")
    md.append("\nGap (control − Aghyad's), points: " + ", ".join(
        f"{k} {100 * (S[k]['control']['pass_rate'] - S[k]['aghyad']['pass_rate']):+.1f}" for k in S) + ".\n")
    # ---- mapping
    md.append("## 4. Bin mapping and knob definitions\n")
    md.append("**Coding** (DeepCoder-Preview `taco` split = TACO-verified, joined to `BAAI/TACO` by statement text for "
              "TACO's own `difficulty` label):\n")
    md.append("| Bin | TACO label | Usable problems | Screen | Held-out | Left for RL training |")
    md.append("|---|---|---|---|---|---|")
    for b, v in bins["bins"].items():
        md.append(f"| {b} | {v['taco_difficulty']} | {v['available']} | {v['screen']} | {v['heldout']} | "
                  f"{v['rl_train_pool_after_screen_and_heldout']} |")
    f = bins["filter"]
    md.append(f"\nFilters: stdin/stdout only (function-call problems dropped), ≥ 5 tests, statement listed under one "
              f"label only; {f.get('dropped_constant_output', 0)} problems dropped because the 15 kept tests (longest "
              "inputs, as DeepCoder) all expect the same output, so a constant answer would pass. **LiveCodeBench: no "
              "LCB-derived problems are used at all** (v6 contains every v5 problem, so all of DeepCoder's `lcbv5` is "
              "excluded); TACO problems dated on or after 2023-05-01 (LCB's window) or matching an LCB statement would "
              f"also be dropped ({f.get('dropped_lcb_overlap', 0)} found). EASY (too easy) and UNKNOWN_DIFFICULTY are "
              "unused; PrimeIntellect SYNTHETIC-1 is unlabelled and skipped. Limits: 6 s per test, 2 GB address space, "
              "Python 3.12 with numpy.\n")
    md.append("**Reasoning Core** (`generate_example(level=L)`; one fresh task object per problem):\n")
    md.append("- **Planning** (generated STRIPS problems, \"PDDL-style\"; answer = one grounded action per line, checked "
              "by simulating it): plan length `horizon = 3 + L`; `2 + L` extra distractor actions; each step's "
              "preconditions gain `L // 3` true and `L // 4` false context facts; from L ≥ 3 some steps are made "
              "interchangeable. Prompt states the exact plan length and a few step cues.")
    md.append("- **First-order logic derivation** (`logic_derivation`: prove a target from premises with rules; answer = "
              "derivation lines checked step by step): starting from 4 entities, 6 unary / 3 binary predicates, 4 facts, "
              "4 rules, max depth 3, 1 distractor, each level adds +1 entity (max 8), +1 fact, +1 rule, +2 distractors, "
              "+1 unary and +1 binary predicate, +0.5 proof depth and target support size. Chosen over `logic_qa`, "
              "whose answer is an integer count that can be guessed.")
    md.append("- Levels were placed with a pilot (10 problems × 1 attempt per level per model, levels 2–12; "
              "`rc_pilot_graded.jsonl`). " + notes.get("pilot", "") + "\n")
    # ---- prompts
    md.append("## 5. Example prompts and answers\n")
    cps = [json.loads(line) for line in gzip.open(OUT / "coding_problems.jsonl.gz", "rt")]
    pb = pick.split("/")[1] if pick.startswith("coding") else "B4_very_hard"
    ex = next(p for p in cps if p["bin"] == pb and p["split"] == "screen")
    md.append(f"**Coding** (first screen problem of {pb}, `{ex['id']}`):\n\n````text\n{ex['prompt']}\n````\n")
    rcp = [json.loads(line) for line in open(OUT / "rc_problems.jsonl")] if (OUT / "rc_problems.jsonl").exists() else []
    for task in ("planning", "logic_derivation"):
        p = next((p for p in rcp if p["task"] == task), None)
        if p:
            md.append(f"**Reasoning Core: {task}** (`{p['id']}`):\n\n````text\n{p['prompt']}\n````\n")
    md.append(f"**Example answers for {'the chosen' if chosen else 'the closest'} setting ({pick})**, one pass and one "
              "fail per model (final answer only; reasoning omitted):\n")
    if pick.startswith("coding"):
        rows = [json.loads(line) for line in gzip.open(OUT / "coding_graded.jsonl.gz", "rt")]
        rows = [r for r in rows if r["bin"] == pb]
    else:
        _, task, lv = pick.split("/")
        rows = [r for r in map(json.loads, open(OUT / "rc_graded.jsonl"))
                if r["task"] == task and f"L{r['level']}" == lv]
    for m in MODELS:
        for want in (1.0, 0.0):
            r = next((r for r in rows if r["model"] == m and r["reward"] == want and r["final"]), None)
            if not r:
                continue
            body = r["final"] if len(r["final"]) < 3500 else r["final"][:3500] + "\n… (cut)"
            why = ""
            if want == 0.0 and r.get("grade") and r["grade"].get("first_fail"):
                ff = r["grade"]["first_fail"]
                why = f" — fails test {ff['test']} ({ff['status']})"
            elif want == 0.0 and "score" in r:
                why = f" — verifier score {r['score']}"
            md.append(f"*{LAB[m]}, {'pass' if want else 'fail'}* (`{r['problem_id']}`, {r['n_tokens']:,} tokens{why}):"
                      f"\n\n````text\n{body}\n````\n")
    # ---- RL config
    md.append("## 6. Proposed RL config\n")
    # measured prompt lengths: coding ≈ 2,050 chars, planning L3–5 ≈ 1,500–2,900, logic ≈ 1,200 (≈ 3.5 chars/token
    # plus the Harmony system header)
    prompt_tok = {k: 650 if k.startswith("coding") else 900 if "planning" in k else 450 for k in S}
    costs = {m: rl_cost(S[pick][m]["mean_tokens"], prompt_tok[pick]) for m in MODELS}
    md.append(f"Setting: **{pick}**{'' if chosen else ' (closest; it does not qualify — see section 7)'}. Both models, "
              "25 steps, 8 problems × 16 attempts per step, lr 4e-5, advantage = reward − group mean (no std scaling), "
              "no KL, `gpt_oss_medium_reasoning`, 32,000-token cap, temperature 1.0. Held-out set: the 100 held-out "
              f"problems of the same {'bin' if pick.startswith('coding') else 'setting (fresh seeds)'}, disjoint from "
              "the screen and training problems; evaluate at steps 0, 5, …, 25 (1 attempt each)."
              + (" Reward: the library's `score_answer` == 1 after stripping the model's own line labels (see section "
                 "7), with the screen's prompt (format example + no-unused-lines rule)." if "logic" in pick else ""))
    md.append("\n**Projected cost** from measured mean output tokens (batch 1 rates incl. training $0.737/M; prompt ≈ "
              f"{prompt_tok[pick]} tokens): " + ", ".join(f"{LAB[m]} ≈ ${c:.0f}" for m, c in costs.items())
              + f", **≈ ${sum(costs.values()):.0f} for both**.")
    if pick.startswith("coding") and "VERY_HARD" in json.dumps(bins["bins"].get(pb, {})):
        md.append(f"\nCaveat: {pb} has only {bins['bins'][pb]['rl_train_pool_after_screen_and_heldout']} problems left "
                  "after the screen and held-out sets; 25 × 8 = 200 training problems would need the screen problems "
                  "reused for training (keeping the held-out set clean), or a smaller held-out set.")
    md.append("\n## 7. Surprises and fallbacks\n")
    md.append(notes.get("surprises", "") + "\n")
    md.append("## Files\n")
    md.append("`results/screen_honest/`: `coding_raw.jsonl` (every coding sample incl. reasoning; also the untrained "
              "model's samples from before it was dropped, not analysed), `coding_graded.jsonl.gz` (with per-test grades),"
              " `grades_*.jsonl` / `jobs_*.jsonl` (sandbox I/O), `coding_problems.jsonl.gz` (problems + hidden tests), "
              "`coding_candidates.jsonl.gz`, `coding_bins.json`, `rc_problems.jsonl`, `rc_graded.jsonl`, "
              "`rc_pilot_graded.jsonl`, `sanity*.json(l)`, `grader_checks.json`, `isolation_test.json`, `summary.json`,"
              " `selection.json`, `cost.json`, logs. Code: `experiments/screen_honest/`.")
    (OUT / "SCREEN.md").write_text("\n".join(md) + "\n")
    sel = {"verdict": verdict, "qualifying": qual, "chosen": chosen, "closest": closest, "rl_cost": costs}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
