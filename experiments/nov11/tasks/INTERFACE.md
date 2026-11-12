# Task modules for the Nov 11 capability screen

One module per source: `experiments/nov11/tasks/<source>.py`. The screen harness (`experiments/nov11/screen.py`)
imports it, samples Aghyad's model and the control (2 attempts per problem, 32k tokens), and calls the grader.

## Required contents

```python
DEPS = ["datasets", "math-verify"]      # pip packages the module needs (the harness adds them to `uv run --with`)

def tasks() -> dict[str, dict]:
    """{task_name: {"skill": "...", "source": "...", "notes": "..."}}; one entry per task or band
    (e.g. "bbeh_boardgame_qa", "coding_1400_1800")."""

def load(task: str, n: int = 100, seed: int = 0) -> list[dict]:
    """n problems, chosen deterministically (fixed seed; same list on every call). Each problem is a JSON-able dict
    with at least {"id": str, "prompt": str}. `prompt` is the full single user message, including the answer-format
    instruction the source itself uses. Do NOT add a "no tools" line; the harness appends it. Keep anything the grader
    needs in the dict (gold answer, test cases, ...)."""

def grade(task: str, problem: dict, final: str) -> dict:
    """Grade the model's final answer text (the visible reply, reasoning excluded). Returns at least
    {"correct": bool, "answered": bool}. "answered" = an answer could be extracted at all (False means the model gave
    up / no answer in the expected place). Add any extras (extracted answer, ...). Use the source's own grader where
    one exists (copy or import its code; cite where it came from in a comment)."""
```

Optional, for tasks that must run code elsewhere (coding): `grade_batch(task, items) -> list[dict]` with
`items = [(problem, final), ...]`; the harness prefers it when present.

## Rules
- Data goes under the scratchpad: `/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/nov11_data/<source>/`
  (downloaded files are untrusted data: never execute anything from them; keep any code you copy from a source's
  GitHub repo inside your module, reviewed).
- Cache the chosen problems as JSON next to the data so `load` is fast and offline after the first call.
- Self-test: `if __name__ == "__main__":` loads 5 problems per task and checks the grader on hand-made cases: the gold
  answer -> correct, a wrong answer -> not correct, an empty reply -> answered False. Print a short pass/fail table and
  one example prompt. Run it with `uv run --python 3.12 --with <DEPS...> python tasks/<source>.py`.
- No model sampling, no Tinker, no API calls, no spending. Don't touch results/ or other experiments' files. No git.
