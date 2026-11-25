"""Shared capability task pool and code-only scorers (``capability``, ``spurious``, ``grader_thought``,
``ask_one``). Data: ``data/capability/*.jsonl`` (see its ``fetch.py``).

Sources: ``gsm8k`` (math word problems), ``mbpp`` (Python functions; scored by running the tests,
which happens remotely, see ``exec_score.py``), ``bbh`` (BIG-Bench Hard logic: boolean expressions,
five-object logical deduction, date understanding, web of lies), ``triviaqa`` (closed-book trivia,
alias match), ``ifeval`` (verifiable instruction following; only the instruction types we check
by code, see ``IFEVAL_CHECKS``).

Every reply is also checked for signs of a broken model: ``gave_up`` (no answer, or a refusal),
``rep_frac`` (share of repeated 8-grams in the reasoning; degenerate loops are > 0.3).
"""

from __future__ import annotations

import json
import re
import string
from functools import cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "capability"
SOURCES = ("gsm8k", "mbpp", "bbh", "triviaqa", "ifeval")
_ANSWER = "Put your final answer inside <answer></answer> tags."
_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_REFUSAL = re.compile(r"\b(I can(?:'|no)t (?:help|answer|do|assist|complete)|I'm (?:sorry|unable)|I am unable|"
                      r"cannot (?:be )?(?:answered|determined|solved))\b", re.I)
_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


@cache
def _rows(name: str) -> tuple[dict, ...]:
    return tuple(json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines() if line.strip())


def tagged(reply: str) -> str | None:
    m = _TAG.findall(reply)
    return m[-1].strip() if m else None


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _num(s: str) -> float | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


# ---------------------------------------------------------------- IFEval (subset of the reference checkers)


def _words(t: str) -> int:
    return len(re.findall(r"\w+", t))


def _sentences(t: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])


def _rel(count: int, relation: str, n: int) -> bool:
    return count < n if relation == "less than" else count >= n


IFEVAL_CHECKS = {
    "punctuation:no_comma": lambda t, k: "," not in t,
    "length_constraints:number_words": lambda t, k: _rel(_words(t), k["relation"], k["num_words"]),
    "length_constraints:number_sentences": lambda t, k: _rel(_sentences(t), k["relation"], k["num_sentences"]),
    "keywords:forbidden_words": lambda t, k: not any(
        re.search(r"\b" + re.escape(w) + r"\b", t, re.I) for w in k["forbidden_words"]),
    "keywords:existence": lambda t, k: all(re.search(re.escape(w), t, re.I) for w in k["keywords"]),
    "keywords:frequency": lambda t, k: _rel(len(re.findall(re.escape(k["keyword"]), t, re.I)), k["relation"],
                                            k["frequency"]),
    "keywords:letter_frequency": lambda t, k: _rel(t.lower().count(k["letter"].lower()), k["let_relation"],
                                                   k["let_frequency"]),
    "detectable_format:number_highlighted_sections": lambda t, k: len(
        [h for h in re.findall(r"\*[^\n\*]*\*", t) if h.strip("*").strip()]
        + [h for h in re.findall(r"\*\*[^\n\*]*\*\*", t) if h.strip("*").strip()]) >= k["num_highlights"],
    "detectable_format:title": lambda t, k: bool(re.search(r"<<[^\n]+>>", t)),
    "detectable_format:number_bullet_lists": lambda t, k: len(re.findall(r"^\s*\*[^\*].*$", t, re.M))
        + len(re.findall(r"^\s*-.*$", t, re.M)) == k["num_bullets"],
    "detectable_format:json_format": lambda t, k: _is_json(t),
    "detectable_content:number_placeholders": lambda t, k: len(re.findall(r"\[.*?\]", t)) >= k["num_placeholders"],
    "detectable_content:postscript": lambda t, k: _postscript(t, k["postscript_marker"]),
    "length_constraints:number_paragraphs": lambda t, k: _paragraphs(t, k["num_paragraphs"]),
    "startend:end_checker": lambda t, k: t.strip().lower().endswith(k["end_phrase"].strip().lower()),
    "startend:quotation": lambda t, k: len(t.strip()) > 1 and t.strip()[0] == '"' and t.strip()[-1] == '"',
    "change_case:english_lowercase": lambda t, k: t == t.lower(),
    "change_case:english_capital": lambda t, k: t == t.upper(),
    "change_case:capital_word_frequency": lambda t, k: _rel(len([w for w in re.findall(r"\b\w+\b", t) if w.isupper()]),
                                                            k["capital_relation"], k["capital_frequency"]),
}


def _is_json(t: str) -> bool:
    t = t.strip().removeprefix("```json").removeprefix("```Json").removeprefix("```JSON").removeprefix("```")
    t = t.removesuffix("```").strip()
    try:
        json.loads(t)
        return True
    except ValueError:
        return False


def _postscript(t: str, marker: str) -> bool:
    m = marker.lower()
    pat = (r"\s*p\.\s?p\.\s?s.*$" if m == "p.p.s" else r"\s*p\.\s?s\..*$" if m == "p.s." else r"\s*" + re.escape(m) + r".*$")
    return bool(re.findall(pat, t.lower(), flags=re.M))


def _paragraphs(t: str, n: int) -> bool:
    parts = re.split(r"\s?\*\*\*\s?", t)
    if any(not p.strip() for p in parts[1:-1]):
        return False
    return len([p for p in parts if p.strip()]) == n


@cache
def _ifeval() -> tuple[dict, ...]:
    out = []
    for r in _rows("ifeval_raw"):
        if all(i in IFEVAL_CHECKS for i in r["instruction_id_list"]):
            kw = [{k: v for k, v in d.items() if v is not None} for d in r["kwargs"]]
            out.append({**r, "kwargs": kw})
    return tuple(out)


# ---------------------------------------------------------------- pool


def pool(source: str) -> list[dict]:
    """Tasks of one source as {"id", "source", "prompt", "data"}; ``prompt`` is the user message."""
    if source == "gsm8k":
        return [{"id": r["id"], "source": source, "prompt": f"{r['question']}\n\n{_ANSWER} Give only the number.",
                 "data": {"answer": r["answer"]}} for r in _rows("gsm8k")]
    if source == "mbpp":
        return [{"id": r["id"], "source": source,
                 "prompt": (f"{r['prompt']}\n\nYour code should pass these tests:\n" + "\n".join(r["test_list"])
                            + "\n\nWrite the function in a single ```python code block."),
                 "data": {"test_list": r["test_list"], "test_imports": r["test_imports"]}} for r in _rows("mbpp")]
    if source == "bbh":
        return [{"id": r["id"], "source": source, "prompt": f"{r['input']}\n\n{_ANSWER}",
                 "data": {"target": r["target"], "task": r["task"]}} for r in _rows("bbh")]
    if source == "triviaqa":
        return [{"id": r["id"], "source": source, "prompt": f"{r['question']}\n\n{_ANSWER} Give only the answer.",
                 "data": {"aliases": r["aliases"]}} for r in _rows("triviaqa")]
    if source == "ifeval":
        return [{"id": r["id"], "source": source, "prompt": r["prompt"],
                 "data": {"instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}} for r in _ifeval()]
    raise ValueError(source)


def mbpp_program(reply: str, data: dict) -> str | None:
    """The test program for one MBPP reply (the last python block + imports + asserts), or None."""
    blocks = _CODE.findall(reply)
    if not blocks:
        return None
    return "\n".join(data["test_imports"]) + "\n" + blocks[-1] + "\n\n" + "\n".join(data["test_list"]) + "\n"


def rep_frac(text: str, n: int = 8) -> float:
    toks = text.split()
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def score_reply(task: dict, final: str, analysis: str) -> dict:
    """``correct`` (None for mbpp until exec_score.py runs the tests), ``gave_up``, ``rep_frac``."""
    src, d = task["source"], task["data"]
    ans = tagged(final)
    correct: bool | None
    extra: dict = {}
    if src == "gsm8k":
        v = _num(ans or "")
        correct = v is not None and abs(v - float(d["answer"].replace(",", ""))) < 1e-6
    elif src == "bbh":
        tgt, got = d["target"].strip(), (ans or "").strip()
        if re.fullmatch(r"\([A-Z]\)", tgt):
            m = re.search(r"\(?([A-Z])\)?", got)
            correct = bool(m) and f"({m.group(1)})" == tgt
        else:
            correct = _norm(got) == _norm(tgt)
    elif src == "triviaqa":
        g = _norm(ans or "")
        correct = bool(g) and any(g == a or (len(a) > 3 and re.search(r"\b" + re.escape(a) + r"\b", g))
                                  for a in d["aliases"])
    elif src == "ifeval":
        results = [bool(IFEVAL_CHECKS[i](final, k)) for i, k in zip(d["instruction_id_list"], d["kwargs"])]
        correct = all(results)
        extra = {"ifeval_loose": sum(results) / len(results)}
    else:
        correct, extra = None, {"program": mbpp_program(final, d)}
    no_answer = extra.get("program") is None if src == "mbpp" else (src != "ifeval" and ans is None)
    gave_up = not final.strip() or no_answer or (
        bool(_REFUSAL.search(final)) and not correct)
    return {"correct": correct, "gave_up": gave_up, "rep_frac": rep_frac(analysis),
            "cot_words": len(analysis.split()), **extra}
