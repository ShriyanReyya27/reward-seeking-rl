"""Dose-response zero-shot: how many scored examples does each arm need to start following a hidden rule?

    uv run --with tinker python experiments/hidden_preference/meeting/c_updating/run.py --stage smoke
    uv run --with tinker python experiments/hidden_preference/meeting/c_updating/run.py --stage probe
    uv run --with tinker python experiments/hidden_preference/meeting/c_updating/run.py --stage main \\
        --rules number:largest,choice:longest,choice:shortest --n 80

Cued framing, a fixed rule per condition, no training. k in {0, 1, 2, 4, 8} scored earlier exchanges, shown as
all rewarded ("all") or half rewarded ("balanced"). Histories are nested: an episode's k = 2 history is the
first 2 of its k = 8 history, and the final request is the same at every k, so a larger k only adds evidence.
Final requests are shared across rules in a family (same index, same request), so the k = 0 prompt is
rule-independent and each k = 0 sample is graded against every rule in the family (``meets``).

Stages share one ``samples.jsonl`` and one spend record. ``probe`` runs k = 0 only, on different requests from
``main`` (seed offset), to pick rules with matched priors. Rerunning skips finished samples and counts their
spend against the cap. The cap is ``TOTAL_CAP_USD`` over all stages (hard limit, enforced here). One JSON
line per sample goes to ``samples.jsonl``; ``cost.json`` holds spend per stage.

Sampling reuses the Tinker harness and API-key loading of ``zero_shot.py`` (``--harness``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

from rsrl.envs import make
from rsrl.envs.base import Task

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from zero_shot import COT_SCORE, default_harness  # noqa: E402

TOTAL_CAP_USD = 12.0  # hard limit across every stage of this experiment, not just one invocation
MAX_TOKENS = 16000  # c430 truncated up to 15% at 8k tokens in earlier runs
KS = (0, 1, 2, 4, 8)
MIXES = ("all", "balanced")
HIST_LEN = max(KS)
SAMPLES_PATH = HERE / "samples.jsonl"
COST_PATH = HERE / "cost.json"
CONFIG_PATH = HERE / "config.json"

SEED_MAIN, SEED_PROBE, SEED_SMOKE = 0, 1, 99


def rule_has(env, rule, info, cands) -> tuple[list, list]:
    pos = [c for c in cands if rule.check(info, env.parse(info, c))]
    neg = [c for c in cands if not rule.check(info, env.parse(info, c))]
    return pos, neg


def pick_final(env, rng, split, rules):
    """An item on this split's side of the hash where every rule in ``rules`` has a passing and a failing valid answer."""
    want = split == "test"
    for _ in range(100_000):
        item = env.make_item(rng)
        if env.in_test(item) != want:
            continue
        cands = env.candidates(item.info, rng)
        if all(all(rule_has(env, r, item.info, cands)) for r in rules):
            return item
    raise RuntimeError(f"{env.family}: no final item fits {[r.name for r in rules]}")


def paid_flags(mix, rng):
    """Which of the 8 history positions scored 1.0. 'balanced': each consecutive pair has one pass and one fail,
    in random order, so every even-length prefix is exactly half rewarded and k = 1 is a coin flip."""
    if mix == "all":
        return [True] * HIST_LEN
    flags = []
    for _ in range(HIST_LEN // 2):
        pair = [True, False]
        rng.shuffle(pair)
        flags += pair
    return flags


def episode_tasks(family, rule_names, env_factory, split, seed, i, ks, mixes):
    """One request index -> tasks at every (rule, mix, k). The final request is shared by every rule of the family."""
    env0 = env_factory(rule_names[0])
    rng = random.Random(f"c_updating:{family}:{split}:{seed}:{i}")
    rules = [next(r for r in env0.rules() if r.name == n) for n in rule_names]
    final = pick_final(env0, rng, split, rules)
    base_id = f"{family}-{split}-{seed}-{i:03d}"
    tasks = []
    if 0 in ks:  # rule-independent prompt, one task per request; every rule is read from ``meets``
        tasks.append((env0, Task(
            id=f"{base_id}-k0-shared", prompt=env0._prompt(final, []),
            metadata={"family": family, "rule": rule_names[0], "shared_k0": True, "k": 0, "mix": "none", "n_pos": 0,
                      "framing": "cued", "nudge": "none", "base_id": base_id, "info": final.info, "n_consistent": None,
                      "consistent": [], "history": []})))
    for rule_name in rule_names:
        env = env_factory(rule_name)
        rule = env._rule(rule_name)
        for mix in mixes:
            hrng = random.Random(f"{base_id}:{rule_name}:{mix}:history")
            flags = paid_flags(mix, hrng)
            used = {final.question}
            hist = []
            for paid in flags:
                item, pos, neg = env._split_item(hrng, split, rule, used)
                hist.append((item, hrng.choice(pos if paid else neg), paid))
            for k in ks:
                if k == 0:
                    continue
                h = hist[:k]
                consistent = [r.name for r in env.rules() if all(r.check(x.info, env.parse(x.info, a)) == p for x, a, p in h)]
                tasks.append((env, Task(
                    id=f"{base_id}-{rule_name}-{mix}-k{k}", prompt=env._prompt(final, h),
                    metadata={"family": family, "rule": rule_name, "shared_k0": False, "k": k, "mix": mix,
                              "n_pos": sum(p for _, _, p in h), "framing": "cued", "nudge": "none", "base_id": base_id,
                              "info": final.info, "n_consistent": len(consistent), "consistent": consistent,
                              "history": [{"info": x.info, "answer": a, "paid": p} for x, a, p in h]})))
    return tasks


def build(stage, design, n, n0, ks, mixes):
    """Returns [(env, task, sample_index), ...]."""
    seed = {"smoke": SEED_SMOKE, "probe": SEED_PROBE, "main": SEED_MAIN}[stage]
    out = []
    for family, rule_names in design.items():
        def factory(rule):
            return make(f"hidden_preference_{family}", k=0, mix="balanced", framing="cued", rule=rule)
        for i in range(n):
            for env, task in episode_tasks(family, rule_names, factory, "test", seed, i, ks, mixes):
                reps = n0 if task.metadata["k"] == 0 else 1
                out += [(env, task, s) for s in range(reps)]
    return out


def load_rows():
    if not SAMPLES_PATH.exists():
        return []
    return [json.loads(l) for l in SAMPLES_PATH.read_text().splitlines() if l.strip()]


def spend_so_far(rows):
    """Larger of the logged sample costs and the per-stage ledger (the ledger also covers billed failures)."""
    ledger = json.loads(COST_PATH.read_text()) if COST_PATH.exists() else {}
    return max(sum(r["cost_usd"] for r in rows), sum(v.get("spent_usd", 0.0) for v in ledger.values()))


def parse_design(spec):
    design = {}
    for item in spec.split(","):
        fam, rule = item.strip().split(":")
        design.setdefault(fam, [])
        if rule not in design[fam]:
            design[fam].append(rule)
    return design


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("smoke", "probe", "main"), required=True)
    ap.add_argument("--rules", default="number:largest,choice:longest,choice:shortest",
                    help="family:rule list; 'main' and 'smoke' run these (rules of one family share final requests)")
    ap.add_argument("--n", type=int, default=None, help="final requests per family (default: 2 smoke, 64 probe, 80 main)")
    ap.add_argument("--n0", type=int, default=4, help="samples per request at k = 0 (rule-independent prompt)")
    ap.add_argument("--budget-usd", type=float, default=TOTAL_CAP_USD, help=f"cap across all stages; never above {TOTAL_CAP_USD}")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--harness", type=Path, default=default_harness())
    a = ap.parse_args()
    cap = min(a.budget_usd, TOTAL_CAP_USD)
    sys.path.insert(0, str(a.harness.resolve()))
    from tinker_client import ARMS, Budget, Sampler  # noqa: E402

    arms = ["c430", "oss430"]
    if a.stage == "probe":
        design = {"choice": ["alpha_first", "alpha_last", "longest", "shortest", "listed_first", "listed_last"],
                  "word": ["starts_vowel", "ends_vowel", "long"],
                  "tagline": ["title_case", "ends_exclaim"]}
        ks, mixes, n, n0 = (0,), (), a.n or 64, 1
    else:
        design = parse_design(a.rules)
        ks, mixes, n, n0 = KS, MIXES, a.n or (2 if a.stage == "smoke" else 80), a.n0
    jobs = build(a.stage, design, n, n0, ks, mixes)
    rows = load_rows()
    done = {(r["stage"], r["arm"], r["task_id"], r["i"]) for r in rows}
    todo = [(arm, e, t, s) for e, t, s in jobs for arm in arms if (a.stage, arm, t.id, s) not in done]
    todo.sort(key=lambda j: zlib.crc32(f"{j[0]}{j[2].id}{j[3]}".encode()))  # interleave, so a budget stop stays balanced
    spent0 = spend_so_far(rows)
    print(f"{a.stage}: {len(jobs)} sample slots, {len(todo)} to run, already spent ${spent0:.2f}, cap ${cap:.2f}, "
          f"max_tokens {a.max_tokens}", flush=True)
    if cfg_new := {"harness": str(a.harness), "arm_paths": {x: ARMS[x] for x in arms}, "ks": list(ks), "mixes": list(mixes),
                   "design": design, "n_requests": n, "n0": n0, "max_tokens": a.max_tokens, "cap": cap, "n_jobs": len(jobs),
                   "seed": {"smoke": SEED_SMOKE, "probe": SEED_PROBE, "main": SEED_MAIN}[a.stage]}:
        cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        cfg[a.stage] = cfg_new
        CONFIG_PATH.write_text(json.dumps(cfg, indent=1))
    if not todo:
        return
    sampler = Sampler(arms, Budget(cap, spent0), a.concurrency)
    await sampler.start()
    f = open(SAMPLES_PATH, "a")
    lock = asyncio.Lock()
    n_done, n_skip, t0 = 0, 0, time.time()

    async def one(arm, env, task, s):
        nonlocal n_done, n_skip
        seed = zlib.crc32(f"{task.id}-{s}".encode()) % 2**31  # same seed for every arm
        c = await sampler.complete(arm, task.prompt, seed=seed, max_tokens=a.max_tokens)
        if c is None:
            n_skip += 1
            return
        g = await env.grade(task, c.parsed.final)
        m = task.metadata
        analysis = c.parsed.analysis
        if not analysis.strip():  # c430 often drops the leading <|channel|> token; recover the block (as rl.py does)
            mm = re.search(r"analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)", c.raw, re.S)
            analysis = mm.group(1) if mm else re.split(r"<\|(?:end|start|channel|return|call)\|>", c.raw)[0]
        rec = {"stage": a.stage, "arm": arm, "task_id": task.id, "i": s, "base_id": m["base_id"], "family": m["family"],
               "rule": m["rule"], "shared_k0": m["shared_k0"], "k": m["k"], "mix": m["mix"], "n_pos": m["n_pos"],
               "n_consistent": m["n_consistent"], "consistent": m["consistent"], "reward": g.reward,
               **{key: g.info[key] for key in ("valid", "meets", "tagged", "answer")},
               "cot_score": bool(COT_SCORE.search(analysis)), "finished": c.parsed.finished,
               "stop_reason": c.stop_reason, "n_prompt_tokens": c.n_prompt, "n_output_tokens": c.n_out,
               "cost_usd": c.cost, "final": c.parsed.final, "analysis": analysis[:6000],
               "analysis_len": len(analysis)}
        async with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(todo)} done, ${sampler.budget.spent:.2f} total, {time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*(one(*j) for j in todo))
    f.close()
    ledger = json.loads(COST_PATH.read_text()) if COST_PATH.exists() else {}
    stage_spend = sampler.budget.spent - spent0
    entry = ledger.get(a.stage, {"spent_usd": 0.0, "samples": 0, "runs": 0})
    entry.update(spent_usd=entry["spent_usd"] + stage_spend, samples=entry["samples"] + n_done, runs=entry["runs"] + 1)
    entry["last_finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry["unit"] = "USD at harness prices (tinker_client PRICE_PREFILL / PRICE_SAMPLE)"
    ledger[a.stage] = entry
    COST_PATH.write_text(json.dumps(ledger, indent=1))
    print(f"finished {n_done} in {time.time() - t0:.0f}s, this stage ${stage_spend:.2f}, experiment total ${sampler.budget.spent:.2f}"
          + (f"; {n_skip} skipped (budget cap or API errors)" if n_skip else ""), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
