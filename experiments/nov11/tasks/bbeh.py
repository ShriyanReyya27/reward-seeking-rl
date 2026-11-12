"""BIG-Bench Extra Hard (BBEH) task module for the Nov 11 capability screen.

Source: https://github.com/google-deepmind/bbeh (Apache-2.0), data at bbeh/benchmark_tasks/<task>/task.json.
Prompt = the task input + the answer-format suffix the BBEH paper appends to every problem
(arXiv:2502.19187, Appendix C). Grading = BBEH's own bbeh/evaluate.py (copied below, unmodified logic).

Chosen tasks (text-only, single answer, varied reasoning; zebra_puzzles excluded, no cryptarithm-like tasks):
  boardgame_qa         - multi-hop deduction with contradictory rules + preferences (proved/disproved/unknown)
  causal_understanding - causal judgement / counterfactual reasoning (Yes/No/Ambiguous)
  time_arithmetic      - date/time calculation with constraints
  word_sorting         - sort words or find the mistaken step in a sorting trace
  multistep_arithmetic - nested arithmetic with newly defined operators (rule following + precision)
"""

import json
import os
import random
import urllib.request

DEPS: list[str] = []

DATA_DIR = ("/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/"
            "0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/nov11_data/bbeh")
REPO_DIR = os.path.join(DATA_DIR, "repo")

_TASKS = {
    "bbeh_boardgame_qa": ("deductive reasoning with defeasible rules and preferences", "proved/disproved/unknown"),
    "bbeh_causal_understanding": ("causal judgement and counterfactual reasoning", "Yes/No/Ambiguous (majority human judgement)"),
    "bbeh_time_arithmetic": ("temporal/date arithmetic", "number, date, list or 'unanswerable'"),
    "bbeh_word_sorting": ("algorithmic: alphabetical sorting / locating an error in a sorting trace", "word list or step number"),
    "bbeh_multistep_arithmetic": ("arithmetic with novel operator definitions", "integer"),
}

# Suffix from the BBEH paper, Appendix C ("added to all problems").
SUFFIX = (
    'Think step by step, and when you provide the final answer, please use the prefix "The answer is:" '
    "without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. "
    'For instance: "The answer is: 42" or "The answer is: yes". If the question is multiple choice with a single '
    "correct answer, the final answer must only be the letter corresponding to the correct answer. "
    'For example, "The answer is: (a)"'
)


def tasks() -> dict[str, dict]:
    return {
        name: {"skill": skill, "source": f"BBEH {name[5:]} (google-deepmind/bbeh)", "notes": f"answer: {ans}; 200 total, 100 sampled"}
        for name, (skill, ans) in _TASKS.items()
    }


def _task_file(task: str) -> str:
    path = os.path.join(REPO_DIR, "bbeh", "benchmark_tasks", task, "task.json")
    if not os.path.exists(path):  # download the raw data file only (no code from the repo is executed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://raw.githubusercontent.com/google-deepmind/bbeh/main/bbeh/benchmark_tasks/{task}/task.json"
        urllib.request.urlretrieve(url, path)
    return path


def load(task: str, n: int = 100, seed: int = 0) -> list[dict]:
    assert task in _TASKS, task
    cache = os.path.join(DATA_DIR, f"{task}_n{n}_s{seed}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    with open(_task_file(task)) as f:
        examples = json.load(f)["examples"]
    idxs = sorted(random.Random(seed).sample(range(len(examples)), min(n, len(examples))))
    probs = [
        {"id": f"{task}_{i}", "prompt": examples[i]["input"].rstrip() + "\n\n" + SUFFIX,
         "answer": examples[i]["target"], "task": task, "source_index": i}
        for i in idxs
    ]
    with open(cache, "w") as f:
        json.dump(probs, f)
    return probs


# ---- Copied from google-deepmind/bbeh, bbeh/evaluate.py (Apache-2.0, Copyright 2025 Google LLC). ----
# Only the module-level example prints were dropped.

def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    """Extracts the final answer from the sample."""
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is "
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Fuzzy match function for BigBench Extra Hard."""
    if prediction == reference:
        return True

    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


def preprocess_sample(sample: str) -> str:
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)

# ---- end of copied code ----

_PREFIXES = ["The answer is:", "The final answer is ", "The final answer is: ", "The answer is "]


def grade(task: str, problem: dict, final: str) -> dict:
    final = final or ""
    prediction = preprocess_sample(final)
    # BBEH's extractor falls back to the whole reply when no prefix is present; we count that as "answered" only if the
    # reply is a single line (i.e. a bare answer), otherwise the model did not answer in the expected place.
    has_prefix = any(p in final for p in _PREFIXES)
    answered = bool(prediction.strip()) and (has_prefix or len(final.strip().splitlines()) == 1)
    return {"correct": evaluate_correctness(final, problem["answer"]), "answered": answered,
            "extracted": prediction, "has_prefix": has_prefix}


if __name__ == "__main__":
    rows = []
    for t in tasks():
        probs = load(t)
        assert len(probs) == 100 and probs == load(t)
        for p in probs[:5]:
            g = grade(t, p, f"Reasoning...\nThe answer is: {p['answer']}")
            w = grade(t, p, "Reasoning...\nThe answer is: zzz wrong 12345")
            e = grade(t, p, "")
            ok = g["correct"] and g["answered"] and not w["correct"] and w["answered"] and not e["answered"] and not e["correct"]
            rows.append((p["id"], g["correct"], w["correct"], e["answered"], "PASS" if ok else "FAIL"))
    print(f"{'id':40s} gold  wrong  empty_answered  result")
    for r in rows:
        print(f"{r[0]:40s} {str(r[1]):5s} {str(r[2]):6s} {str(r[3]):15s} {r[4]}")
    print(f"{sum(r[4] == 'PASS' for r in rows)}/{len(rows)} passed")
    print("\n--- example prompt ---\n" + load("bbeh_causal_understanding")[0]["prompt"])
