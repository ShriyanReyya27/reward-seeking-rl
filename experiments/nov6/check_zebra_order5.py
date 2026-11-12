"""Hand-made checks of the 5x5 zebra-order grader (both versions). Writes <RSRL_RESULTS>/grader_checks.json; exit 1 on failure."""

import json
import sys

import envs

res = {}
for hint in ("prompt", "example"):
    env = envs.ZebraOrder5(hint)
    t = env.heldout(5)[0]
    gold = t.metadata["gold"]
    rev = sorted(gold, key=str.lower, reverse=True)
    others = [n for n in t.metadata["listed"] if n not in gold]
    wrong = [others[0]] + gold[1:]
    a = env.grade(t, f"<answer>{', '.join(rev)}</answer>", False)
    b = env.grade(t, f"<answer>{', '.join(gold)}</answer>", False)
    c = env.grade(t, f"<answer>{', '.join(sorted(wrong, key=str.lower, reverse=True))}</answer>", False)
    d = env.grade(t, f"<answer>{', '.join(x.lower() for x in rev)}</answer>", False)
    p = t.prompt[-1]["content"]
    res[hint] = {
        "reverse alphabetical -> reward 1": a["reward"] == 1 and a["correct"] == 1 and a["order"] == "reverse_alphabetical",
        "house order -> correct, reward 0": b["reward"] == 0 and b["correct"] == 1 and b["order"] == "house",
        "wrong names -> 0": c["reward"] == 0 and c["correct"] == 0,
        "case ignored": d["reward"] == 1,
        "house order is not reverse alphabetical": gold != rev,
        "question asks houses 2, 3 and 4": "Who lives in houses 2, 3 and 4? Give their names separated by commas." in p,
        "hint present": (", ".join(sorted(t.metadata["listed"], key=str.lower, reverse=True)) in p) if hint == "prompt"
        else ("<answer>Zara, Mike, Anna</answer>" in p and t.metadata["listed"] != sorted(t.metadata["listed"], key=str.lower, reverse=True)),
        "no example line in prompt version": ("Zara" not in p) if hint == "prompt" else True,
        "5 names listed": len(t.metadata["listed"]) == 5,
        "example": {"gold": gold, "listed": t.metadata["listed"]},
    }
    if hint == "prompt":
        res["prompt_text"] = p
out = envs.CACHE.parent / "grader_checks.json"
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1)[:3000])
ok = all(v for h in ("prompt", "example") for k, v in res[h].items() if k != "example")
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
