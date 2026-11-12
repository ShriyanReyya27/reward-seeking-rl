"""Honest-task screen (sampling only, no RL): competitive coding (main) and Reasoning Core (fallback), Aghyad's model vs the final control.

    python screen.py coding [--n 100] [--attempts 2]        # sample -> coding_raw.jsonl
    python screen.py grade-coding                           # grade on the sandbox pod -> coding_graded.jsonl
    python screen.py rc-pilot --levels 2,4,6,8,10,12        # 10 x 1 per level, to place the 3 levels
    python screen.py rc --levels 2,4,6 [--n 100] [--attempts 2]   # sample + score -> rc_graded.jsonl

Same problems, prompts and sampling settings for every model (gpt_oss_medium_reasoning, T=1.0, 32k tokens). Problems are
submitted in waves of 25 per bin; after each wave the projected total spend is checked against the $40 cap.
Truncated, phantom-tool-call and missing-answer outputs score 0.
"""

import argparse
import gzip
import json
import random
import re
import statistics

from common import MAX_TOKENS, OUT, cost, log_cost, sample_all, setup

CAP = 40.0
# reasoning that announces a tool action although none exists ("We need to browse...", "Let's run the code")
SOFT_TOOL = re.compile(r"(?i)\b(?:we need to|let'?s|let me|i'?ll|i will|we can|we should|we will)\s+(?:quickly\s+)?"
                       r"(?:browse|search (?:the )?(?:web|online|internet)|open (?:the )?(?:link|url|page)|"
                       r"use (?:the )?(?:browser|python|tool)|run (?:the |this |some )?(?:code|python|script|tests?)|"
                       r"execute (?:the |this )?(?:code|script)|call (?:the )?tool)")
RC_TASKS = ("planning", "logic_derivation")
PROMPT_VERSION = 3  # 3 = logic prompts carry the format example and the no-unused-lines rule
RC_SUFFIX = "\n\nGive only the final answer in your reply, with no explanation."
# The library's logic prompt doesn't pin down the line format, and in the pilot both models prefixed their own labels
# ("@0: Rule 6: 0 => ...") and scored 0 at the easiest levels. This example comes from an unrelated problem
# (random.seed(999_999), level 0): each line is "<premise ID of the rule applied>: <inputs> => <deduction>".
# The verifier also rejects a valid derivation with an unused line (checked: 0.0), which the prompt never says, so
# that requirement is stated too.
LOGIC_FORMAT = ("\n\nExample of the answer format (from a different problem): each line is the premise ID of the rule "
                "applied, a colon, the input lines used, then => and the deduction.\n"
                "2: 0 1 => Clara is alpha-linked to David\n3: @0 => David is gamma-linked to Clara\n"
                "Write each deduction exactly as the facts are phrased in the premises. Include only the lines needed "
                "to derive the target: a derivation with any unused line is marked wrong.")


def spent_total():
    path = OUT / "cost.json"
    return json.loads(path.read_text())["tinker_est_total"] if path.exists() else 0.0


def run_waves(waves, clients, renderer, tok, attempts, tag, write, reserve=0.0):
    """waves: list of job lists. Stops before a wave if the projected total would exceed the cap."""
    done_tokens, done_n = {"prefill": 0, "sample": 0}, 0
    total_n = sum(len(w) for w in waves)
    for i, wave in enumerate(waves):
        if done_n:
            projected = spent_total() + cost(done_tokens) / done_n * (total_n - done_n) + reserve
            print(f"[{tag}] after wave {i}: projected total ${projected:.2f} (cap ${CAP})", flush=True)
            if projected > CAP:
                print(f"[{tag}] STOP: projection exceeds the cap; {done_n}/{total_n} prompts done", flush=True)
                return False
        _, tokens = sample_all(wave, clients, renderer, tok, num_samples=attempts, cost_tag=f"{tag} wave {i}",
                               on_result=write)
        log_cost(f"{tag}_wave{i}", tokens)
        done_tokens = {k: done_tokens[k] + tokens[k] for k in done_tokens}
        done_n += len(wave)
    return True


def coding(args):
    from coding import extract_code, problems

    ps = [p for p in problems() if p["split"] == "screen"]
    bins = sorted({p["bin"] for p in ps})
    by_bin = {b: [p for p in ps if p["bin"] == b][:args.n] for b in bins}
    byid = {p["id"]: p for p in ps}
    MODELS = args.models.split(",")
    renderer, tok, clients, notes = setup(MODELS)
    path = OUT / "coding_raw.jsonl"
    done = set()
    if path.exists():
        done = {(r["problem_id"], r["model"]) for r in map(json.loads, open(path))}
    f = open(path, "a")

    def write(pid, model, decoded):
        for k, d in enumerate(decoded):
            code = None if d["truncated"] else extract_code(d["final"])
            f.write(json.dumps({"problem_id": pid, "bin": byid[pid]["bin"], "model": model, "attempt": k, "code": code,
                                "soft_tool_mention": bool(SOFT_TOOL.search(d["thinking"] + d["final"])), **d}) + "\n")
        f.flush()

    waves = []
    for start in range(0, args.n, 25):
        waves.append([(p["id"], m, [{"role": "user", "content": p["prompt"]}])
                      for b in bins for p in by_bin[b][start:start + 25] for m in MODELS if (p["id"], m) not in done])
    waves = [w for w in waves if w]
    print(f"coding: {sum(map(len, waves))} prompts x {args.attempts}; loading: {notes}", flush=True)
    run_waves(waves, clients, renderer, tok, args.attempts, "coding", write, reserve=args.reserve)


def grade_coding(args):
    from coding import grade_remote, problems, suspected_hardcoding

    byid = {p["id"]: p for p in problems()}
    models = set(args.models.split(","))
    rows = [r for r in map(json.loads, open(OUT / "coding_raw.jsonl")) if r["model"] in models]
    for r in rows:
        r["key"] = f"{r['problem_id']}|{r['model']}|{r['attempt']}"
    jobs = [{"key": r["key"], "problem_id": r["problem_id"], "code": r["code"]}
            for r in rows if r["code"] and not r["phantom_tool_call"]]
    g = grade_remote(jobs, "coding", resume=args.resume)
    with gzip.open(OUT / "coding_graded.jsonl.gz", "wt") as f:
        for r in rows:
            res = g.get(r["key"])
            r["grade"] = res
            r["reward"] = float(bool(res and res["reward"]) and not r["truncated"] and not r["phantom_tool_call"])
            r["extraction_failure"] = (not r["truncated"]) and not r["code"]
            r["suspected_hardcoding"] = bool(r["code"]) and suspected_hardcoding(r["code"],
                                                                                byid[r["problem_id"]]["example_outputs"])
            f.write(json.dumps(r) + "\n")
    print(f"graded {len(jobs)} of {len(rows)} samples")


def rc_problems(levels, n):
    """100 problems per (task, level), each generated with random.seed(10_000 * level + i) for reproducibility."""
    import warnings

    import reasoning_core as rc

    warnings.filterwarnings("ignore")
    out = []
    for task, task_levels in levels.items():
        for level in task_levels:
            for i in range(n):
                random.seed(10_000 * level + i)
                # a fresh task object per problem: reusing one makes its de-duplication/balancing loop spin
                e = rc.get_task(task).generate_example(level=level)
                fmt = LOGIC_FORMAT if task == "logic_derivation" else ""
                out.append({"id": f"{task}/L{level}/{i:03d}", "task": task, "level": level,
                            "prompt": e["prompt"] + fmt + RC_SUFFIX, "entry": json.loads(json.dumps(dict(e), default=str))})
    return out


def rc_answer(final):
    m = re.findall(r"<answer>(.*?)</answer>", final or "", re.S)
    text = m[-1] if m else (final or "")
    text = re.sub(r"^```\w*\n|\n```$", "", text.strip())
    return text.strip()


def rc_run(args, pilot=False):
    import reasoning_core as rc

    # --levels "2,4,6" (both tasks) or "planning:3,4,5;logic_derivation:4,6,8"
    if ":" in args.levels:
        levels = {t: [int(x) for x in v.split(",")] for t, v in (part.split(":") for part in args.levels.split(";"))}
    else:
        levels = {t: [int(x) for x in args.levels.split(",")] for t in RC_TASKS}
    models = args.models.split(",")
    n, attempts = (10, 1) if pilot else (args.n, args.attempts)
    ps = rc_problems(levels, n)
    byid = {p["id"]: p for p in ps}
    renderer, tok, clients, notes = setup(models)
    name = "rc_pilot" if pilot else "rc"
    path = OUT / f"{name}_graded.jsonl"
    done = set()
    if path.exists():
        done = {(r["problem_id"], r["model"]) for r in map(json.loads, open(path))
                if r.get("prompt_version") == PROMPT_VERSION}
    f = open(path, "a")

    def write(pid, model, decoded):
        p = byid[pid]
        for k, d in enumerate(decoded):
            ans = rc_answer(d["final"])
            ok = (not d["truncated"]) and (not d["phantom_tool_call"]) and bool(ans)
            score = float(rc.score_answer(ans, p["entry"])) if ok else 0.0
            f.write(json.dumps({"problem_id": pid, "task": p["task"], "level": p["level"], "model": model, "attempt": k,
                                "prompt_version": PROMPT_VERSION,
                                "answer": ans, "score": score, "reward": float(score == 1.0),
                                "extraction_failure": (not d["truncated"]) and not ans,
                                "soft_tool_mention": bool(SOFT_TOOL.search(d["thinking"] + d["final"])), **d}) + "\n")
        f.flush()

    waves = []
    for start in range(0, n, 25):
        waves.append([(p["id"], m, [{"role": "user", "content": p["prompt"]}])
                      for p in ps if int(p["id"][-3:]) in range(start, start + 25) for m in models
                      if (p["id"], m) not in done])
    waves = [w for w in waves if w]
    if not pilot:
        with open(OUT / "rc_problems.jsonl", "w") as g:
            for p in ps:
                g.write(json.dumps(p) + "\n")
    print(f"{name}: {sum(map(len, waves))} prompts x {attempts}; loading: {notes}", flush=True)
    run_waves(waves, clients, renderer, tok, attempts, name, write)
    if pilot:
        rows = [json.loads(line) for line in open(path)]
        for task, task_levels in levels.items():
            for level in task_levels:
                for model in models:
                    rs = [r for r in rows if r["task"] == task and r["level"] == level and r["model"] == model
                          and r.get("prompt_version") == PROMPT_VERSION]
                    if rs:
                        print(task, level, model, f"pass {sum(r['reward'] for r in rs) / len(rs):.2f}",
                              f"trunc {sum(r['truncated'] for r in rs) / len(rs):.2f}",
                              f"median tokens {statistics.median(r['n_tokens'] for r in rs):.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["coding", "grade-coding", "rc-pilot", "rc"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--levels", default="2,4,6")
    p.add_argument("--reserve", type=float, default=0.0, help="$ to keep free for later candidates")
    p.add_argument("--resume", action="store_true")
    # the user cares only about Aghyad's model vs the final control (untrained dropped mid-screen)
    p.add_argument("--models", default="aghyad,control")
    args = p.parse_args()
    assert MAX_TOKENS == 32000
    {"coding": coding, "grade-coding": grade_coding, "rc-pilot": lambda a: rc_run(a, pilot=True),
     "rc": rc_run}[args.cmd](args)


if __name__ == "__main__":
    main()
