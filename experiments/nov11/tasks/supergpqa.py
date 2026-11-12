"""SuperGPQA (m-a-p/SuperGPQA), hard subset (difficulty == "hard"), zero-shot.

Prompt and answer extraction mirror the official repo (github.com/SuperGPQA/SuperGPQA):
  - prompt: config/prompt/zero-shot.yaml + infer/data_loader.py (options formatted "A) ...")
  - grading: eval/eval.py, zero-shot mode: extract_option_labels(response, 'ABCDEFGHIJ'); if None,
    fall back to extract_option_content(response, options); correct iff predict == answer_letter.
"""
import json
import random
import re
from pathlib import Path

DEPS = ["datasets"]

DATA_DIR = Path(
    "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/"
    "0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/nov11_data/supergpqa"
)
POOL = DATA_DIR / "hard_pool.json"
TASK = "supergpqa_hard"

# Verbatim from SuperGPQA config/prompt/zero-shot.yaml
PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. There is only one correct answer. The last line of your "
    "response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, "
    "F, G, H, I, or J.\n\n{}"
)


def tasks() -> dict[str, dict]:
    return {
        TASK: {
            "skill": "graduate-level multi-discipline knowledge/reasoning (multiple choice, up to 10 options)",
            "source": "m-a-p/SuperGPQA (HF), train split, difficulty == 'hard' (7050 problems)",
            "notes": "Official zero-shot prompt and official answer extraction (eval/eval.py). 100 problems chosen by "
            "seeded shuffle with proportional (largest-remainder) stratification over `discipline`, so the mix "
            "follows the hard subset (~60% Science, ~35% Engineering, a few Medicine/Law/Economics).",
        }
    }


def _pool() -> list[dict]:
    if POOL.exists():
        return json.loads(POOL.read_text())
    from datasets import load_dataset

    ds = load_dataset("m-a-p/SuperGPQA", split="train", cache_dir=str(DATA_DIR / "hf"))
    rows = [
        {k: r[k] for k in ("uuid", "question", "options", "answer", "answer_letter", "discipline", "field", "subfield")}
        for r in ds
        if r["difficulty"] == "hard"
    ]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    POOL.write_text(json.dumps(rows))
    return rows


def _prompt(r: dict) -> str:
    # infer/data_loader.py, zero-shot mode
    body = r["question"] + "\n" + "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(r["options"]))
    return PROMPT_TEMPLATE.format(body)


def load(task: str, n: int = 100, seed: int = 0) -> list[dict]:
    assert task == TASK, task
    cache = DATA_DIR / f"chosen_n{n}_s{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pool = sorted(_pool(), key=lambda r: r["uuid"])
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for r in pool:
        groups.setdefault(r["discipline"], []).append(r)
    for d in sorted(groups):
        rng.shuffle(groups[d])
    # largest-remainder proportional allocation across disciplines
    total = len(pool)
    quotas = {d: n * len(g) / total for d, g in groups.items()}
    alloc = {d: int(q) for d, q in quotas.items()}
    for d in sorted(quotas, key=lambda d: (-(quotas[d] - alloc[d]), d))[: n - sum(alloc.values())]:
        alloc[d] += 1
    chosen = [r for d in sorted(groups) for r in groups[d][: alloc[d]]]
    rng.shuffle(chosen)
    out = [
        {
            "id": f"supergpqa_{r['uuid']}",
            "prompt": _prompt(r),
            "answer_letter": r["answer_letter"],
            "answer": r["answer"],
            "options": r["options"],
            "discipline": r["discipline"],
            "field": r["field"],
            "subfield": r["subfield"],
        }
        for r in chosen
    ]
    cache.write_text(json.dumps(out, indent=1))
    return out


# ---- Copied from SuperGPQA eval/eval.py (extract_option_labels / extract_option_content). ----
# Changes: patterns rewritten as raw f-strings (checked identical to the originals' compiled strings), and
# the timeout_decorator-wrapped safe_regex_search replaced by a plain re.search.
def _search(pattern, text, flags=0):
    try:
        return re.search(pattern, text, flags)
    except Exception:
        return None


def extract_option_labels(text, options="ABCDEFGHIJ"):
    if not isinstance(text, str) or not isinstance(options, str):
        return "error"
    text = text.rstrip()
    last_line = text.split("\n")[-1]
    option_str = "".join([chr(65 + i) for i in range(len(options))]) if options else "ABCDEFGHIJ"
    patterns = [
        # e.g. "The final answer to this question is: A." / "The best option is $\boxed{B}:" / "The correct answer is (C)."
        rf"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*(?:[\*\$\{{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*([{option_str}])(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)",
        # e.g. "ANSWER: A" / "Answer: $\boxed{B}." / "ANSWER: (C):"
        rf"(?i:Answer)[\*\s]*:\s*(?:[\*\$\{{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*([{option_str}])(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)",
        # e.g. "A" / "$\boxed{B}$" / "(C)." / "[D]:"
        rf"^[^\w" "\r\n" rf"]*(?:[\*\$\{{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*([{option_str}])(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)",
    ]
    for pattern in patterns:
        match = _search(pattern, last_line, re.IGNORECASE)
        if match:
            return match.group(1)
    for pattern in patterns:
        match = _search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_option_content(text, options_content=None):
    if not isinstance(text, str) or not isinstance(options_content, list):
        return "error"
    escaped_options_content = [re.escape(option_content) for option_content in options_content]
    escaped_options_content_str = "|".join(escaped_options_content)
    text = text.rstrip()
    last_line = text.split("\n")[-1]
    patterns = [
        rf"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*(?:[\*\$\{{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*({escaped_options_content_str})(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)",
        rf"(?i:Answer)\s*(?:[\*\$\{{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*({escaped_options_content_str})(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)",
        rf"^[^\w" "\r\n" rf"]*(?:[\*\$\{{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*({escaped_options_content_str})(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)",
    ]
    for pattern in patterns:
        match = _search(pattern, last_line)
        if match:
            if match.group(1) in escaped_options_content:
                return options_content[escaped_options_content.index(match.group(1))]
            return match.group(1)
    for pattern in patterns:
        match = _search(pattern, text)
        if match:
            if match.group(1) in escaped_options_content:
                return options_content[escaped_options_content.index(match.group(1))]
            return match.group(1)
    return None


# ---- end copied code ----


def grade(task: str, problem: dict, final: str) -> dict:
    final = final or ""
    pred = extract_option_labels(final, "ABCDEFGHIJ")
    via = "label"
    if pred is None:
        content = extract_option_content(final, problem["options"])
        via = "content"
        # eval.py compares answer_letter to whatever was extracted; a matched option text never equals a letter,
        # so we map it to its letter (otherwise the content fallback could never be marked correct).
        if content is not None and content in problem["options"]:
            pred = chr(65 + problem["options"].index(content))
        else:
            pred = content
    answered = pred not in (None, "error")
    return {
        "correct": bool(answered and pred == problem["answer_letter"]),
        "answered": answered,
        "extracted": pred,
        "extracted_via": via,
    }


if __name__ == "__main__":
    probs = load(TASK)
    print(f"{TASK}: pool={len(_pool())} chosen={len(probs)}")
    from collections import Counter

    print("disciplines:", dict(Counter(p["discipline"] for p in probs)))
    assert [p["id"] for p in load(TASK)] == [p["id"] for p in probs]
    ok_all = True
    print(f"{'id':44s} gold  wrong  empty  boxed  content")
    for p in probs[:5]:
        g = p["answer_letter"]
        wrong = "A" if g != "A" else "B"
        r_gold = grade(TASK, p, f"Some reasoning.\nAnswer: {g}")
        r_wrong = grade(TASK, p, f"Some reasoning.\nAnswer: {wrong}")
        r_empty = grade(TASK, p, "")
        r_box = grade(TASK, p, f"So the answer is $\\boxed{{{g}}}$.")
        r_content = grade(TASK, p, f"Thinking...\nThe correct answer is {p['answer']}")
        row = [
            r_gold["correct"] and r_gold["answered"],
            not r_wrong["correct"] and r_wrong["answered"],
            not r_empty["answered"] and not r_empty["correct"],
            r_box["correct"],
            r_content["correct"],
        ]
        ok_all &= all(row)
        print(f"{p['id']:44s} " + "  ".join("PASS" if x else "FAIL" for x in row))
    print("ALL PASS" if ok_all else "SOME FAILED")
    print("\n--- example prompt ---\n" + probs[0]["prompt"])
