"""Omni-MATH (KbsdJames/Omni-MATH), hardest tier (difficulty >= 9.0, i.e. the 9.0 and 9.5 levels).

Omni-MATH's official grader is a model judge (Omni-Judge / GPT-4o), which we can't use here (no sampling).
Instead: extract the last \\boxed{...} from the reply and check it against the gold answer with math-verify
(parse + verify), falling back to normalized string equality. Problems whose gold answer can't be checked
this way (empty, > 60 chars, or proof/prose-like) are dropped.
"""
import json
import random
import re
import threading
from pathlib import Path

DEPS = ["datasets", "math-verify"]

DATA_DIR = Path(
    "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/"
    "0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/nov11_data/omnimath"
)
POOL = DATA_DIR / "hardest_pool.json"
TASK = "omnimath_hardest"
MIN_DIFFICULTY = 9.0
MAX_ANSWER_LEN = 60
PROSE_RE = re.compile(
    r"proof|prove|show that|there (exist|is|are)|infinitely many|converge|diverge|sufficiently|explanation|see solution",
    re.I,
)
INSTRUCTION = "Put your final answer in \\boxed{}."


def tasks() -> dict[str, dict]:
    return {
        TASK: {
            "skill": "olympiad-level competition math with exact final answers",
            "source": "KbsdJames/Omni-MATH (HF), test split, difficulty >= 9.0 (levels 9.0 and 9.5; 173 problems)",
            "notes": "Threshold difficulty >= 9.0 (>= 8.0 would add ~280 easier problems; 9.x alone leaves 136 "
            "checkable). Dropped 37 of 173: empty, longer than 60 chars, or proof/prose-like "
            f"(regex {PROSE_RE.pattern!r}). Grader: last \\boxed{{}} vs gold via math-verify parse+verify, "
            "fallback normalized string equality (official Omni-MATH grading is an LLM judge, not used).",
        }
    }


def _clean_gold(a: str) -> str:
    a = a.strip()
    a = re.sub(r"^\$+|\$+$", "", a).strip()
    b = last_boxed(a)
    if b is not None and a.startswith("\\boxed"):
        a = b
    return a.strip()


def _checkable(a: str) -> bool:
    return bool(a) and len(a) <= MAX_ANSWER_LEN and not PROSE_RE.search(a)


def _pool() -> tuple[list[dict], dict]:
    if POOL.exists():
        d = json.loads(POOL.read_text())
        return d["rows"], d["stats"]
    from datasets import load_dataset

    ds = load_dataset("KbsdJames/Omni-MATH", split="test", cache_dir=str(DATA_DIR / "hf"))
    hard, rows = 0, []
    for i, r in enumerate(ds):
        if r["difficulty"] < MIN_DIFFICULTY:
            continue
        hard += 1
        gold = _clean_gold(r["answer"])
        if _checkable(gold):
            rows.append(
                {"idx": i, "problem": r["problem"], "gold": gold, "raw_answer": r["answer"],
                 "difficulty": r["difficulty"], "domain": r["domain"], "source": r["source"]}
            )
    stats = {"at_threshold": hard, "checkable": len(rows), "dropped": hard - len(rows)}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    POOL.write_text(json.dumps({"rows": rows, "stats": stats}))
    return rows, stats


def load(task: str, n: int = 100, seed: int = 0) -> list[dict]:
    assert task == TASK, task
    cache = DATA_DIR / f"chosen_n{n}_s{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rows, _ = _pool()
    rows = sorted(rows, key=lambda r: r["idx"])
    random.Random(seed).shuffle(rows)
    out = [
        {
            "id": f"omnimath_{r['idx']}",
            "prompt": r["problem"].strip() + "\n\n" + INSTRUCTION,
            "gold": r["gold"],
            "raw_answer": r["raw_answer"],
            "difficulty": r["difficulty"],
            "domain": r["domain"],
            "source": r["source"],
        }
        for r in rows[:n]
    ]
    cache.write_text(json.dumps(out, indent=1))
    return out


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...} (nested braces handled), or `\\boxed x` without braces."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    j = idx + (6 if text.startswith("\\boxed", idx) else 5)
    if j < len(text) and text[j] == " ":
        m = re.match(r" +([^\s$]+)", text[j:])
        return m.group(1) if m else None
    if j >= len(text) or text[j] != "{":
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def _norm(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^{}]*)\}|\\textbf\{([^{}]*)\}|\\mathrm\{([^{}]*)\}",
               lambda m: next(g for g in m.groups() if g is not None), s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\:|\\quad|\\qquad", "", s)
    s = re.sub(r"\s+", "", s).rstrip(".").lower()
    return s


def _math_equal(gold: str, pred: str) -> bool:
    from math_verify import parse, verify

    # math-verify's timeouts use SIGALRM, which only works on the main thread.
    t = 5 if threading.current_thread() is threading.main_thread() else None
    try:
        g = parse("\\boxed{" + gold + "}", parsing_timeout=t)
        p = parse("\\boxed{" + pred + "}", parsing_timeout=t)
        return bool(g and p and verify(g, p, timeout_seconds=t))
    except Exception:
        return False


def grade(task: str, problem: dict, final: str) -> dict:
    pred = last_boxed(final or "")
    if pred is None or not pred.strip():
        return {"correct": False, "answered": False, "extracted": pred}
    gold = problem["gold"]
    by_mv = _math_equal(gold, pred)
    by_str = _norm(gold) == _norm(pred)
    return {"correct": by_mv or by_str, "answered": True, "extracted": pred, "math_verify": by_mv, "str_eq": by_str}


if __name__ == "__main__":
    probs = load(TASK)
    _, stats = _pool()
    print(f"{TASK}: {stats} chosen={len(probs)}")
    assert [p["id"] for p in load(TASK)] == [p["id"] for p in probs]
    ok_all = True
    print(f"{'id':16s} gold  wrong  empty  nobox  | gold answer")
    for p in probs[:5]:
        r_gold = grade(TASK, p, f"Reasoning...\nThe answer is $\\boxed{{{p['gold']}}}$.")
        r_wrong = grade(TASK, p, "Reasoning...\n\\boxed{1234567}")
        r_empty = grade(TASK, p, "")
        r_nobox = grade(TASK, p, f"The answer is {p['gold']}.")
        row = [r_gold["correct"], r_wrong["answered"] and not r_wrong["correct"],
               not r_empty["answered"] and not r_empty["correct"], not r_nobox["answered"]]
        ok_all &= all(row)
        print(f"{p['id']:16s} " + "  ".join("PASS" if x else "FAIL" for x in row) + f"  | {p['gold']}")
    # extra checks: equivalent forms
    extra = [("\\frac{1}{2}", "0.5", True), ("2n(n - 1) + 1", "2n^2-2n+1", True), ("80 + 40\\sqrt{3}", "3", False),
             ("\\text{yes}", "Yes", True), ("q \\in \\{-2, 0, 2\\}", "\\{0, 2, -2\\}", True)]
    for g, pr, want in extra:
        got = grade(TASK, {"gold": g}, f"\\boxed{{{pr}}}")["correct"]
        ok_all &= got == want
        print(f"equiv {g!r} vs {pr!r}: {'PASS' if got == want else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    print("\n--- example prompt ---\n" + probs[0]["prompt"])
