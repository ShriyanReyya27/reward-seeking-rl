"""Screen 2: word sorting with a stated custom alphabet (problems generated in the BBEH word-sorting style).

Vocabulary: every lowercase word in the "List:" parts of BBEH word_sorting (google-deepmind/bbeh, task.json).
Problem i: 10-20 distinct words drawn with Random(f"sort-{i}"). Screen problems: i = 0..99. Training pool (kept
apart, never screened): i = 10000 and up. Every problem's gold order differs from normal alphabetical order.
Versions:
  a  shuffled alphabet, stated (each problem its own random letter order, Random(f"alpha-{i}"))
  b  fixed rule, stated: vowels first (a, e, i, o, u), then the consonants in normal order
  c  fixed rule NOT stated: the normal sorting prompt, graded with rule (b)
Grading: exact match of the whole sorted list (from the last "Answer:" line).
"""
import json
import random
import re
import string
from collections import Counter

import cn

VERSIONS = ("a", "b", "c")
VOWEL_FIRST = "aeiou" + "".join(c for c in string.ascii_lowercase if c not in "aeiou")
FMT = ("\n\nEnd your response with a final line of the form \"Answer: \" followed by the sorted words, separated by "
       "single spaces.")
PREFIX = " Compare words letter by letter using this order; if one word is the start of another, the shorter word comes first."
_vocab = None


def vocab():
    global _vocab
    if _vocab is None:
        src = cn.DATA.parent / "nov11_data" / "bbeh" / "repo" / "bbeh" / "benchmark_tasks" / "bbeh_word_sorting" / "task.json"
        words = set()
        for ex in json.load(open(src))["examples"]:
            for m in re.finditer(r"List:\s*([a-z' ]+)", ex["input"]):
                words |= {w for w in m.group(1).split() if w.isalpha() and len(w) >= 3}
        _vocab = sorted(words)
    return _vocab


def key_for(alpha):
    rank = {c: i for i, c in enumerate(alpha)}
    return lambda w: [rank[c] for c in w]


SETS = {"base": (10, 20), "hard": (40, 60), "medium": (25, 35)}  # medium added: hard split the models  # hard: added because base was >90% for both models


def make(i, version, set_="base"):
    v = vocab()
    rng = random.Random(f"sort-{i}" if set_ == "base" else f"sort-{set_}-{i}")
    words = rng.sample(v, rng.randint(*SETS[set_]))
    if version == "a":
        alpha = list(string.ascii_lowercase)
        random.Random(f"alpha-{i}").shuffle(alpha)
        alpha = "".join(alpha)
        prompt = (f"Sort the following words using this alphabet order (from first to last): {', '.join(alpha)}."
                  f"{PREFIX}\nList: {' '.join(words)}{FMT}")
    elif version == "b":
        alpha = VOWEL_FIRST
        prompt = ("Sort the words using this alphabet: vowels first (a, e, i, o, u), then the consonants in normal "
                  f"order.{PREFIX}\nList: {' '.join(words)}{FMT}")
    else:
        alpha = VOWEL_FIRST
        prompt = f"Sort the following words alphabetically: List: {' '.join(words)}{FMT}"
    gold = sorted(words, key=key_for(alpha))
    return {"id": f"sort_{version}_{i}" if set_ == "base" else f"sort_{version}_{set_}_{i}", "set": set_, "prompt": prompt, "words": words, "alphabet": alpha, "gold": gold,
            "normal": sorted(words)}


def problems(version, n=100, set_="base", start=0):
    out = []
    i = start
    while len(out) < n:
        p = make(i, version, set_)
        if p["gold"] != p["normal"]:  # skip the rare list whose rule order equals normal order
            out.append(p)
        i += 1
    return out


def parse(final):
    lines = [l for l in (final or "").strip().splitlines() if l.strip()]
    for l in reversed(lines):
        m = re.search(r"answer\s*:\s*(.*)$", l.replace("*", ""), re.I)
        if m:
            toks = re.split(r"[\s,]+", m.group(1).strip())
            return [t.strip(".;:'\"`()[]").lower() for t in toks if t.strip(".;:'\"`()[]")]
    return None


def grade(version, p, final):
    got = parse(final)
    if got is None:
        return {"correct": False, "answered": False, "normal_order": False, "same_set_wrong_order": False,
                "extracted": None}
    correct = got == p["gold"]
    normal = (not correct) and got == p["normal"]
    return {"correct": correct, "answered": True, "normal_order": normal,
            "same_set_wrong_order": (not correct) and (not normal) and Counter(got) == Counter(p["words"]),
            "extracted": got}


if __name__ == "__main__":
    print("vocab", len(vocab()))
    for v in VERSIONS:
        ps = problems(v)
        p = ps[0]
        ok = grade(v, p, "blah\nAnswer: " + " ".join(p["gold"]))["correct"]
        nm = grade(v, p, "**Answer:** " + ", ".join(p["normal"]) + ".")
        sw = grade(v, p, "Answer: " + " ".join(reversed(p["gold"])))
        em = grade(v, p, "")
        print(v, "gold->", ok, "normal->", nm["normal_order"], not nm["correct"], "reversed->", sw["same_set_wrong_order"],
              "empty->", not em["answered"], "| sizes", min(len(x["words"]) for x in ps), max(len(x["words"]) for x in ps))
    print(problems("a", 1)[0]["prompt"]); print(); print(problems("b", 1)[0]["prompt"]); print(); print(problems("c", 1)[0]["prompt"])
    tr = {tuple(sorted(make(i, "a")["words"])) for i in range(10000, 10500)}
    print("overlap screen/training word sets:", sum(tuple(sorted(x["words"])) in tr for x in problems("a")))
