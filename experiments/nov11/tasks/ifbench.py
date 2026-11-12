"""IFBench (allenai/IFBench, HF `allenai/IFBench_test`): single-turn precise instruction following, zero-shot.

Prompt = the dataset prompt as-is. Grading = IFBench's own checker (copied, reviewed, into `ifbench_lib/` from
github.com/allenai/IFBench; see ifbench_lib/__init__.py), STRICT mode: `evaluation_lib.test_instruction_following_strict`.
"correct" = prompt-level strict accuracy (all instructions followed); extra "frac_followed" = instruction-level share.
"""
import copy
import json
import os
import random
import urllib.request
from pathlib import Path

DEPS = ["nltk", "langdetect", "absl-py", "immutabledict", "emoji", "syllapy"]

DATA_DIR = Path("/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/"
                "0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/nov11_data/ifbench")
RAW_FILE = DATA_DIR / "IFBench_test.jsonl"
# Same file as HF allenai/IFBench_test (300 prompts); the repo's data/ copy is the one its README evaluates against.
RAW_URL = "https://raw.githubusercontent.com/allenai/IFBench/main/data/IFBench_test.jsonl"
os.environ.setdefault("NLTK_DATA", str(DATA_DIR / "nltk_data"))  # must be set before ifbench_lib imports nltk

_lib = None


def _evaluation_lib():
    global _lib
    if _lib is None:
        try:
            from ifbench_lib import evaluation_lib  # harness runs with tasks/ on sys.path
        except ImportError:
            from .ifbench_lib import evaluation_lib
        _lib = evaluation_lib
    return _lib


def tasks() -> dict[str, dict]:
    return {"ifbench": {"skill": "precise instruction following (out-of-distribution verifiable constraints)",
                        "source": "allenai/IFBench (IFBench_test, single-turn)",
                        "notes": "strict prompt-level accuracy via IFBench's checker; frac_followed = instruction-level"}}


def _all_rows() -> list[dict]:
    if not RAW_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, RAW_FILE)
    with open(RAW_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]


def load(task: str, n: int = 100, seed: int = 0) -> list[dict]:
    assert task == "ifbench", task
    cache = DATA_DIR / f"{task}_n{n}_s{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rows = sorted(_all_rows(), key=lambda r: int(r["key"]))
    chosen = random.Random(seed).sample(rows, min(n, len(rows)))
    probs = [{"id": f"ifbench_{r['key']}", "prompt": r["prompt"], "key": r["key"],
              "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]} for r in chosen]
    cache.write_text(json.dumps(probs))
    return probs


def grade(task: str, problem: dict, final: str) -> dict:
    lib = _evaluation_lib()
    final = final or ""
    inp = lib.InputExample(key=problem["key"], instruction_id_list=list(problem["instruction_id_list"]),
                           prompt=problem["prompt"], kwargs=copy.deepcopy(problem["kwargs"]))
    error = None
    try:
        out = lib.test_instruction_following_strict(inp, {problem["prompt"]: final})
        followed = list(out.follow_instruction_list)
    except Exception as e:  # a checker crashing on odd output counts as not followed (per instruction)
        error = f"{type(e).__name__}: {e}"
        followed = []
        for i, iid in enumerate(problem["instruction_id_list"]):
            try:
                one = lib.InputExample(key=problem["key"], instruction_id_list=[iid], prompt=problem["prompt"],
                                       kwargs=[copy.deepcopy(problem["kwargs"][i])])
                followed.append(lib.test_instruction_following_strict(one, {problem["prompt"]: final})
                                .follow_instruction_list[0])
            except Exception:
                followed.append(False)
    return {"correct": bool(followed) and all(followed), "answered": bool(final.strip()),
            "frac_followed": sum(followed) / len(followed) if followed else 0.0,
            "follow_instruction_list": followed, "instruction_id_list": problem["instruction_id_list"],
            **({"checker_error": error} if error else {})}


# ---------------------------------------------------------------- self-test
# Hand-written compliant replies, keyed by instruction id; used where the sampled problem's instructions all have one.
def _compliant(problem: dict) -> str | None:
    parts = []
    for iid, kw in zip(problem["instruction_id_list"], problem["kwargs"]):
        kw = {k: v for k, v in kw.items() if v is not None}
        if iid == "count:keywords_multiple":
            words = [kw[f"keyword{i}"] for i in range(1, 6)]
            parts.append(" ".join(f"{w}" for w, c in zip(words, [1, 2, 3, 5, 7]) for _ in range(c)))
        else:
            return None
    return " ".join(parts)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    probs = load("ifbench")
    print(f"loaded {len(probs)} problems (pool {len(_all_rows())}); ids: {[p['id'] for p in probs[:5]]}")
    rows = []
    for p in probs[:5]:
        e, w = grade("ifbench", p, ""), grade("ifbench", p, "ok")
        rows.append((p["id"], ",".join(p["instruction_id_list"]), "PASS" if not e["answered"] and not e["correct"] else "FAIL",
                     "PASS" if w["answered"] and not w["correct"] else "FAIL"))
    print(f"{'id':<14} {'empty->unanswered':<18} {'\"ok\"->incorrect':<16} instructions")
    for r in rows:
        print(f"{r[0]:<14} {r[2]:<18} {r[3]:<16} {r[1]}")

    # Compliant replies written by hand for specific dataset prompts (by key in the full 300-prompt pool).
    pool = {r["key"]: r for r in _all_rows()}
    hand = {
        "0": _compliant(pool["0"]),  # count:keywords_multiple (kaleidoscope x1, nebula x2, whisper x3, ...)
        "20": "Communities need 3 things: aligned resources, 2 kinds of training, and 1 shared plan.",  # exactly 3 numbers
        "261": "Keep relationships resilient through trust, honesty, and steady mutual support.",  # starts with a verb (NLTK tags "Keep" VB; note "Build"/"Make" get NNP)
        "132": ("* Memorization: the model reproduces training examples verbatim.\n"
                "  - Example: regurgitating a copyrighted paragraph.\n"
                "* Why care: privacy, copyright, and poor generalization.\n"
                "  - Leaked personal data can be extracted by prompting.\n"),  # format:sub-bullets
    }
    noncompliant = {"20": "There are 4, 5, 6 or 7 key needs.", "261": "The relationships matter.",
                    "132": "* Memorization is copying.\n* It matters.\n"}
    print("\nhand-made replies (compliant -> correct; non-compliant -> incorrect):")
    for k, reply in hand.items():
        g = grade("ifbench", {"id": f"ifbench_{k}", **pool[k]}, reply)
        bad = grade("ifbench", {"id": f"ifbench_{k}", **pool[k]}, noncompliant[k]) if k in noncompliant else None
        print(f"  key {k:<4} {str(pool[k]['instruction_id_list']):<30} compliant: {'PASS' if g['correct'] else 'FAIL'}"
              + (f"   non-compliant: {'PASS' if not bad['correct'] else 'FAIL'}" if bad else ""))
    print("\nexample prompt:\n" + probs[0]["prompt"][:1200])
