"""Train/test overlap check for every batch 3 env (run on the training pod; coding tasks come from the sandbox).

    python experiments/batch3/check_splits.py > results/batch3/splits.json

Compares the exact training items each run will see with the held-out eval items: prompt text for zebra and
crypt_hack, and visible + hidden case inputs for the coding tasks.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import KINDS, Coding, Crypt, Zebra  # noqa: E402


def text(t):
    return t.prompt[0]["content"] if hasattr(t, "prompt") else t["prompt"][0]["content"]


def main():
    out = {}
    for name, env, kind in (("E3f", Zebra("E3f", False), "zebra"), ("E4e", Crypt(False), "crypt")):
        k = KINDS[kind]
        train = [t for b in env.train_tasks(k["steps"], k["per_step"]) for t in b]
        test = env.eval_tasks(k["n_eval"])
        overlap = {text(t) for t in train} & {text(t) for t in test}
        out[name] = {"train_items": len(train), "test_items": len(test), "overlap": len(overlap)}
    try:
        env = Coding("S1pen", "http://127.0.0.1:8765", False)
        k = KINDS["S1pen"]
        train = [t for b in env.train_tasks(k["steps"], k["per_step"]) for t in b]
        test = env.eval_tasks(k["n_eval"])
        cases = lambda ts: {json.dumps(c[0], sort_keys=True) for t in ts  # noqa: E731
                            for c in t["metadata"]["visible"] + t["metadata"]["hidden"]}
        out["E5"] = {"train_items": len(train), "test_items": len(test),
                     "overlap_prompts": len({text(t) for t in train} & {text(t) for t in test}),
                     "overlap_case_inputs": len(cases(train) & cases(test))}
    except Exception as error:  # sandbox not reachable
        out["E5"] = {"error": f"{type(error).__name__}: {error}"}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
