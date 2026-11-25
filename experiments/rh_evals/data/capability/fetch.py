"""Fetch the capability-eval subsets from the Hugging Face datasets server into slim JSONL files here.

    python experiments/rh_evals/data/capability/fetch.py

Rows are taken from the start of each test split (the datasets server serves rows in split order), so
the subsets are fixed and reproducible. The outputs are committed; rerun only to change the sizes.
Sources and licences: GSM8K (openai/gsm8k, MIT), MBPP sanitized (google-research-datasets/mbpp,
CC-BY-4.0), BIG-Bench Hard (lukaemon/bbh, MIT), TriviaQA rc.nocontext validation
(mandarjoshi/trivia_qa, Apache-2.0), IFEval (google/IFEval, Apache-2.0).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://datasets-server.huggingface.co/rows"
BBH_TASKS = ("boolean_expressions", "logical_deduction_five_objects", "date_understanding", "web_of_lies")


def rows(dataset: str, config: str, split: str, n: int) -> list[dict]:
    out = []
    while len(out) < n:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": len(out), "length": min(100, n - len(out))})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                    batch = [x["row"] for x in json.load(r)["rows"]]
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"failed to fetch {dataset}/{config}/{split}")
        if not batch:
            break
        out.extend(batch)
    return out


def write(name: str, recs: list[dict]) -> None:
    with open(HERE / f"{name}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(recs)}")


def main() -> None:
    write("gsm8k", [{"id": f"gsm8k-{i}", "question": r["question"], "answer": r["answer"].split("####")[-1].strip()}
                    for i, r in enumerate(rows("openai/gsm8k", "main", "test", 200))])
    write("mbpp", [{"id": f"mbpp-{r['task_id']}", "prompt": r["prompt"], "code": r["code"],
                    "test_list": r["test_list"], "test_imports": r.get("test_imports") or []}
                   for r in rows("google-research-datasets/mbpp", "sanitized", "test", 150)])
    bbh = []
    for t in BBH_TASKS:
        bbh += [{"id": f"bbh-{t}-{i}", "task": t, "input": r["input"], "target": r["target"]}
                for i, r in enumerate(rows("lukaemon/bbh", t, "test", 40))]
    write("bbh", bbh)
    write("triviaqa", [{"id": f"triviaqa-{r['question_id']}", "question": r["question"],
                        "aliases": sorted(set(r["answer"]["normalized_aliases"]) | {r["answer"]["normalized_value"]})}
                       for r in rows("mandarjoshi/trivia_qa", "rc.nocontext", "validation", 200)])
    write("ifeval_raw", [{"id": f"ifeval-{r['key']}", "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
                          "kwargs": r["kwargs"]} for r in rows("google/IFEval", "default", "train", 541)])


if __name__ == "__main__":
    main()
