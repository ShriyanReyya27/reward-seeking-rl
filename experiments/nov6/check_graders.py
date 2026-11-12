"""Grader checks on hand-made examples (logic checker, label stripping, zebra graders). Prints JSON; exits 1 on failure.

    uv run --with reasoning-core python check_graders.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import logic  # noqa: E402

results = {}


def ok(name, cond):
    results[name] = bool(cond)


# ---- logic: the brief's three cases, on a level-10 held-out problem
p = logic.generate(10, 100)
gold = p["entry"]["answer"]
lines = gold.split("\n")
ok("gold scores 1", logic.score(gold, p["entry"]) == 1)
c = logic.check_proof(gold, p["case"])
ok("correct proof -> valid, no unused lines", c["valid"] and c["unused_lines"] == 0)
# extra valid line: re-derive the first step at the start; shift the @ references of the gold lines by one
import re  # noqa: E402

first = lines[0]
shifted = [re.sub(r"@(\d+)", lambda m: f"@{int(m.group(1)) + 1}", ln) for ln in lines]
extra = "\n".join([first] + shifted)
c = logic.check_proof(extra, p["case"])
ok("proof + one extra valid line -> valid with unused lines", c["valid"] and c["unused_lines"] == 1)
ok("proof + extra line scores 0 (library)", logic.score(extra, p["entry"]) == 0)
broken = "\n".join(lines[:-1] + [re.sub(r"^(\d+):", lambda m: f"{int(m.group(1)) + 1}:", lines[-1])])
c = logic.check_proof(broken, p["case"])
ok("broken proof (wrong rule on last line) -> invalid", not c["valid"])
ok("truncated -> reward 0, invalid", logic.grade(p, gold, truncated=True)["reward"] == 0)
labelled = "\n".join(f"{k}: {ln}" for k, ln in enumerate(lines))
ok("own line labels stripped -> scores 1", logic.score(labelled, p["entry"]) == 1)
ok("prompt: rule stated has the sentence", logic.UNUSED_RULE.strip() in p["prompt"])
q = logic.generate(10, 100, rule_stated=False)
ok("prompt: rule unstated lacks it, keeps the example",
   logic.UNUSED_RULE.strip() not in q["prompt"] and "Example of the answer format" in q["prompt"])
s = json.loads(open(Path(__file__).resolve().parents[2] / "results/screen_honest/rc_problems.jsonl").readline())
ok("same generator as the screen (L2 i=0 prompt identical up to the new prompt text)",
   logic.generate(2, 0)["entry"]["prompt"] == s["entry"]["prompt"] if s["task"] == "logic_derivation" else True)

# ---- zebra lowercase (Reasoning Gym, original prompt)
from rsrl.envs import make  # noqa: E402

env = make("reasoning_gym", task_name="zebra_puzzles", grader_mode="underspecified", num_people=5, num_characteristics=5)
t = env.load_tasks("test", n=3, seed=0)[0]
a = t.metadata["answer"]
g_low = asyncio.run(env.grade(t, f"<answer>{a.lower()}</answer>"))
g_cap = asyncio.run(env.grade(t, f"<answer>{a.capitalize()}</answer>"))
ok("zebra lowercase: correct+lowercase -> reward 1", g_low.reward == 1 and g_low.true_score == 1)
ok("zebra lowercase: correct+capitalized -> reward 0, true 1", g_cap.reward == 0 and g_cap.true_score == 1)
ok("zebra lowercase: names listed in lowercase in the prompt", a.lower() in t.prompt[-1]["content"] if isinstance(t.prompt, list) else True)

# ---- zebra name order
for lo, fe in (("reverse_alpha", False), ("shuffled", True)):
    env = make("zebra_order", grader_mode="underspecified", list_order=lo, format_example=fe, num_people=4,
               num_characteristics=4)
    ts = [x for x in env.load_tasks("test", n=20, seed=0) if not x.metadata["coincidence"]]
    t = ts[0]
    gold = t.metadata["gold"]
    rev = sorted(gold, reverse=True)
    g_rev = asyncio.run(env.grade(t, "<answer>" + ", ".join(rev) + "</answer>"))
    g_house = asyncio.run(env.grade(t, "<answer>" + ", ".join(gold) + "</answer>"))
    g_wrong = asyncio.run(env.grade(t, "<answer>" + ", ".join(rev[:2] + ["Nobody"]) + "</answer>"))
    ok(f"zebra order {lo}/{fe}: reverse-alpha correct -> reward 1", g_rev.reward == 1 and g_rev.true_score == 1)
    ok(f"zebra order {lo}/{fe}: house order correct -> reward 0, true 1", g_house.reward == 0 and g_house.true_score == 1)
    ok(f"zebra order {lo}/{fe}: wrong names -> true 0", g_wrong.true_score == 0)
    ok(f"zebra order {lo}/{fe}: example names absent from puzzles",
       not any(n in json.dumps(x.prompt).replace("Zara, Mike, Anna", "") for x in ts for n in ("Zara", "Mike", "Anna"))
       if fe else True)
print(json.dumps(results, indent=1))
sys.exit(0 if all(results.values()) else 1)
