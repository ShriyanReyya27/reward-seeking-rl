"""Logic-puzzle tasks from SynLogic (MiniMaxAI/SynLogic on HF; github.com/MiniMax-AI/SynLogic).

Source data: `synlogic_hard/train.parquet` (the "hard" split). Within it we keep English prompts only (about half are
in Chinese) and take the hardest available items:
  - synlogic_sudoku:    9x9, difficulty 3 only (highest level; sudoku is the only one of these types with levels 1-3)
  - synlogic_calcudoko: KenKen; all 60 English 6x6 + 40 of the 5x5 (6x6 is the largest)
  - synlogic_kukurasu:  6x6 only (largest available)
(Calcudoko/Kukurasu rows all carry difficulty=1, so the hard split + largest grid is the hardness setting.)
Futoshiki was tried and dropped: in synlogic_hard the stored constraints contradict the stored answer for 100/100 of
the English 5x5 items and 84/100 are unsolvable, so the data is broken.
Prompt = SynLogic's own training prompt (the <think>/<answer> wrapper + the puzzle text with its answer format).
Grading = SynLogic's own verifiers (copied into puzzles_lib/synlogic_verifiers.py), applied to the answer extracted the
way SynLogic's RL reward does (docs/reward_example.py: <answer>...</answer> content if present, else whole reply).
We do NOT apply the strict reward-format gate (reply must start with <think>), since the harness grades only the
visible reply.
"""
import contextlib
import io
import json
import os
import random
import re
import urllib.request

DEPS = ["numpy", "pandas", "pyarrow"]

DATA_DIR = ("/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/"
            "0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/nov11_data/puzzles")
PARQUET_URL = "https://huggingface.co/datasets/MiniMaxAI/SynLogic/resolve/main/synlogic_hard/train.parquet"

_TASKS = {
    "synlogic_sudoku": ("sudoku", "Sudoku 9x9, SynLogic difficulty 3 (highest)"),
    "synlogic_calcudoko": ("calcudoko", "Calcudoko/KenKen 6x6 (60) + 5x5 (40): Latin square with arithmetic cages"),
    "synlogic_kukurasu": ("kukurasu", "Kukurasu 6x6: place 1s so weighted row/column sums match"),
}


def tasks() -> dict[str, dict]:
    return {name: {"skill": "logic puzzle (constraint satisfaction)",
                   "source": f"SynLogic synlogic_hard/train.parquet, data_source={src}",
                   "notes": notes + "; English prompts only; graded by SynLogic's own verifier (any valid grid passes)"}
            for name, (src, notes) in _TASKS.items()}


def _size(src, meta):
    return {"kukurasu": lambda: meta["n"], "sudoku": lambda: 9}.get(src, lambda: meta["grid_size"])()


def _build_cache(task, path):
    import pandas as pd
    pq = os.path.join(DATA_DIR, "hard_train.parquet")
    if not os.path.exists(pq):
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(PARQUET_URL, pq)
    src = _TASKS[task][0]
    df = pd.read_parquet(pq)
    df = df[df.data_source == src]
    rows = []
    for _, r in df.iterrows():
        prompt = r.prompt[0]["content"]
        if re.search("[一-鿿]", prompt):
            continue
        game_data_str = r.extra_info["game_data_str"]
        g = json.loads(game_data_str)
        if src == "sudoku" and g["difficulty"] != 3:
            continue
        rows.append({"id": f"{task}_{r.extra_info['index']}", "prompt": prompt, "game_data": game_data_str,
                     "size": _size(src, g["metadata"])})
    rows.sort(key=lambda x: x["id"])
    random.Random(0).shuffle(rows)
    rows.sort(key=lambda x: -x["size"])  # stable: largest grids first, shuffled within size
    with open(path, "w") as f:
        json.dump(rows[:100], f)


def load(task: str, n: int = 100, seed: int = 0) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{task}.json")
    if not os.path.exists(path):
        _build_cache(task, path)
    with open(path) as f:
        rows = json.load(f)
    if n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def _answered(src, ans_text, verifier):
    if src == "sudoku":
        return verifier.extract_answer(ans_text) != ""
    if src == "calcudoko":
        return re.search(r"\[\[(.*?)\]\]", ans_text) is not None  # the verifier's primary extraction path
    if src == "kukurasu":
        return verifier.extract_answer(ans_text) != ""
    raise KeyError(src)


def _sv():
    import importlib.util
    import sys
    name = "_nov11_synlogic_verifiers"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "puzzles_lib", "synlogic_verifiers.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
    return sys.modules[name]


def grade(task: str, problem: dict, final: str) -> dict:
    sv = _sv()
    src = _TASKS[task][0]
    verifier = {"sudoku": sv.SudokuVerifier, "calcudoko": sv.CalcudokoVerifier,
                "kukurasu": sv.KukurasuVerifier}[src]()
    data = sv.Data.from_json_str(problem["game_data"])
    final = final or ""
    ans_text = sv.reward_extract_solution_with_thought(final)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        answered = bool(final.strip()) and _answered(src, ans_text, verifier)
        try:
            correct = bool(verifier.verify(data, ans_text))
        except Exception:
            correct = False
    answered = answered or correct
    return {"correct": correct, "answered": answered, "extracted": ans_text[-500:],
            "has_answer_tag": "<answer>" in final}


# ---------------------------------------------------------------- self-test
def _gold_and_wrong(task, problem):
    g = json.loads(problem["game_data"])
    src = _TASKS[task][0]
    if src == "sudoku":
        grid = [list(r) for r in g["metadata"]["complete_sudoku"]]
    elif src == "calcudoko":
        grid = [[int(x) for x in row.split()] for row in g["answer"].strip("[]").split(",")]
    else:
        sol = g["metadata"]["solution"]
        wrong = [list(r) for r in sol]
        wrong[0][0] = "1" if wrong[0][0] == "X" else "X"
        return json.dumps(sol), json.dumps(wrong)
    wrong = [list(r) for r in grid]
    row = wrong[0]
    j = next(j for j in range(1, len(row)) if row[j] != row[0])
    row[0], row[j] = row[j], row[0]  # swap two cells: row stays a permutation, columns break
    if src == "sudoku":
        fmt = lambda gr: "```python\n" + str(tuple(tuple(r) for r in gr)) + "\n```"
    else:
        fmt = lambda gr: "[[" + ",".join(" ".join(str(x) for x in r) for r in gr) + "]]"
    return fmt(grid), fmt(wrong)


if __name__ == "__main__":
    print(f"{'task':22s} {'n':>4s} {'sizes':18s} {'gold':>6s} {'wrong':>6s} {'empty':>6s} {'gold@100':>9s}")
    ok_all = True
    for task in _TASKS:
        probs = load(task)
        assert len(probs) == 100 and load(task)[0]["id"] == probs[0]["id"]
        sizes = {}
        for p in probs:
            sizes[p["size"]] = sizes.get(p["size"], 0) + 1
        res = {"gold": 0, "wrong": 0, "empty": 0}
        for p in probs[:5]:
            gold, wrong = _gold_and_wrong(task, p)
            rg = grade(task, p, f"Some reasoning.\n<answer>{gold}</answer>")
            rw = grade(task, p, f"<answer>{wrong}</answer>")
            re_ = grade(task, p, "")
            res["gold"] += rg["correct"] and rg["answered"]
            res["wrong"] += (not rw["correct"]) and rw["answered"]
            res["empty"] += not re_["answered"] and not re_["correct"]
        # also: gold without <answer> tags, on all 100 (checks the verifiers accept every reference)
        gold100 = sum(grade(task, p, _gold_and_wrong(task, p)[0])["correct"] for p in probs)
        ok = res == {"gold": 5, "wrong": 5, "empty": 5} and gold100 == 100
        ok_all &= ok
        print(f"{task:22s} {len(probs):4d} {str(sizes):18s} {res['gold']:>4d}/5 {res['wrong']:>4d}/5 "
              f"{res['empty']:>4d}/5 {gold100:>6d}/100  {'PASS' if ok else 'FAIL'}")
    print("\nExample prompt (synlogic_kukurasu):\n" + load("synlogic_kukurasu")[0]["prompt"])
    print("\nALL PASS" if ok_all else "\nSOME FAILED")
