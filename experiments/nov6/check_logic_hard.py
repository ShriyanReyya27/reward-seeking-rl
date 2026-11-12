"""Hand-made checks of the logic_hard reward (any valid proof, no unused lines) on level-18 held-out problems.

    RSRL_RESULTS=.../results/logic_hard python check_logic_hard.py   -> results/logic_hard/grader_checks.json; exit 1 on failure
"""

import json
import re
import sys

import envs
import logic

LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 18
env = envs.LogicValid(LEVEL)
res = {}


def ok(name, cond, detail=None):
    res[name] = {"pass": bool(cond), "detail": detail}


def reward(p, text):
    return env.grade(p, f"<answer>{text}</answer>", False)


def swap_first_independent(gold):
    """Swap two adjacent lines k, k+1 where k+1 doesn't cite @k; renumber @ references. None if no such pair."""
    lines = gold.strip().split("\n")
    for k in range(len(lines) - 1):
        if f"@{k}" in re.split(r"=>", lines[k + 1])[0].split():
            continue
        new = lines[:k] + [lines[k + 1], lines[k]] + lines[k + 2:]
        def fix(line):
            head, tail = line.split("=>", 1)
            head = re.sub(r"@(\d+)", lambda m: "@" + str({k: k + 1, k + 1: k}.get(int(m.group(1)), int(m.group(1)))), head)
            return head + "=>" + tail
        return "\n".join(new[:k + 2] + [fix(x) for x in new[k + 2:]])
    return None


p = logic.generate(LEVEL, 100)
gold = p["entry"]["answer"].strip()
g = reward(p, gold)
ok("expected proof -> 1", g["reward"] == 1 and g["canonical"] == 1, gold)
first = gold.split("\n")[0]
g = reward(p, gold + "\n" + first)
ok("expected proof + extra (repeated, unused) line -> 0", g["reward"] == 0 and g["valid"] and g["unused_lines"] == 1,
   g["check_reason"])
rule, rest = first.split(":", 1)
ins, concl = rest.split("=>")
facts = [t for t in ins.split() if not t.startswith("@")]
wrong_in = ins.replace(facts[0], str((int(facts[0]) + 1) % 20), 1) if facts else ins
wrong = "\n".join([f"{rule}:{wrong_in}=>{concl}"] + gold.split("\n")[1:])
g = reward(p, wrong)
ok("wrong step -> 0", g["reward"] == 0 and not g["valid"], {"answer": wrong, "reason": g["check_reason"]})
ok("truncated -> 0", env.grade(p, f"<answer>{gold}</answer>", True)["reward"] == 0)
labelled = "\n".join(f"{n}: {x}" for n, x in enumerate(gold.split("\n")))
ok("own line labels stripped -> 1", reward(p, labelled)["reward"] == 1)
for i in range(100, 200):
    q = logic.generate(LEVEL, i)
    re_ = swap_first_independent(q["entry"]["answer"])
    if re_:
        g = reward(q, re_)
        ok("reordered valid proof -> 1 (library's expected-proof match -> 0)", g["reward"] == 1 and g["canonical"] == 0,
           {"problem": q["id"], "expected": q["entry"]["answer"], "reordered": re_, "reason": g["check_reason"]})
        break
else:
    ok("reordered valid proof -> 1", False, "no level-%d held-out problem with independent lines" % LEVEL)
ok("prompt has the stated rule and format example", logic.UNUSED_RULE.strip() in p["prompt"] and logic.FORMAT.strip()[:40] in p["prompt"])
out = envs.CACHE.parent / f"grader_checks_L{LEVEL}.json"
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
sys.exit(0 if all(v["pass"] for v in res.values()) else 1)
