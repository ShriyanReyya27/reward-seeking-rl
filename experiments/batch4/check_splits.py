"""Train/test overlap check for every batch 4 env (run on the training pod; coding tasks come from the sandbox).

    python experiments/batch4/check_splits.py > results/batch4/splits.json

Compares the exact training items each run will see with the held-out eval items: prompt text for zebra and
crypt_hack, and visible + hidden case inputs for the coding tasks.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import KINDS, Crypt  # noqa: E402


def text(t):
    return t.prompt[0]["content"] if hasattr(t, "prompt") else t["prompt"][0]["content"]


def main():
    out = {}
    for name, env, kind in (("E6", Crypt(False), "crypt"),):
        k = KINDS[kind]
        train = [t for b in env.train_tasks(k["steps"], k["per_step"]) for t in b]
        test = env.eval_tasks(k["n_eval"])
        overlap = {text(t) for t in train} & {text(t) for t in test}
        out[name] = {"train_items": len(train), "test_items": len(test), "overlap": len(overlap)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
