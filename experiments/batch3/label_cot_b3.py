"""CoT labels for batch 3's E3-flip runs (E3fa, E3fp), reusing results/zebra_cot/label_cot.py unchanged.

    python experiments/batch3/label_cot_b3.py sample    # results/batch3/cot/sample.jsonl
    python experiments/batch3/label_cot_b3.py label     # gpt-4.1 with LABELER_PROMPT_v2.md -> labels_gpt41_v2.jsonl

Same sampling scheme as the zebra CoT analysis: per run, 50 answered training chains from each step bucket
0-4, 5-9, 10-14, 15-19 plus 100 from the whole run (fixed seed). Same prompt (the latest approved, v2, copied
into the output dir so label_cot reads it from there). The only change is the puzzle shown to the labeler:
batch 3's E3-flip prompt (names capitalized, plus the lowercase instruction). Spend cap: $8.
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "results" / "zebra_cot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_cot as L  # noqa: E402

L.OUT = REPO / "results" / "batch3" / "cot"
L.RUNS = {
    "E3fa": ("Aghyad's", "E3-flip", "results/batch3/E3fa/rollouts.jsonl"),
    "E3fp": ("partial control", "E3-flip", "results/batch3/E3fp/rollouts.jsonl"),
}
L.CAP = 8.0
_prompts = {}


def question(task_id, run=None):
    """Batch 3's E3-flip prompt, abridged like label_cot.question (clues dropped)."""
    from train import KINDS, Zebra

    exp = "E3f"
    if exp not in _prompts:
        env = Zebra(exp, False)
        _prompts[exp] = {t.id: t.prompt[0]["content"] for b in env.train_tasks(KINDS["zebra"]["steps"], 8) for t in b}
    q = _prompts[exp][task_id]
    head = q.split("\n\n1.")[0]
    tail = q[q.index("What is"):]
    return f"{head}\n\n[...numbered clues omitted...]\n\n{tail}"


def user_message(template, c):
    # label_cot.user_message looks up label_cot.question at call time; point it at this chain's run first.
    L.question = lambda task_id: question(task_id, c["run"])
    return _user_message(template, c)


_user_message = L.user_message
L.user_message = user_message

if __name__ == "__main__":
    L.OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "results" / "zebra_cot" / "LABELER_PROMPT_v2.md", L.OUT / "LABELER_PROMPT_v2.md")
    cmd = sys.argv[1]
    if cmd == "sample":
        L.sample()
    elif cmd == "label":
        ids = [json.loads(line)["id"] for line in open(L.OUT / "sample.jsonl")]
        asyncio.run(L.label(L.MAIN, ids, L.OUT / "labels_gpt41_v2.jsonl", version=2))
