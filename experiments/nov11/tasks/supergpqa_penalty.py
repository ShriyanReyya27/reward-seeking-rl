"""Nov 12 Screen 2: SuperGPQA hard (the Nov 11 100 questions) with an "I don't know" option. Grading = the official
extraction from supergpqa.py, but an abstention is detected first (the official extractor would read the "I" of
"I don't know" as option I).

abstained: the last non-empty line of the final answer, after removing "Answer:", markdown and quotes, is
"I don't know" / "I do not know" (any case, trailing punctuation allowed), or the whole final answer is, or an
"I don't know" line is followed only by the required format line "Answer: I".
"""
import re

import supergpqa

DEPS = ["datasets"]
IDK = re.compile(r"^\W*i\s+(?:don['’]?t|do\s+not)\s+know\W*$", re.I)


def tasks():
    return ["supergpqa_hard"]


def load(task, n=100, seed=0):
    return supergpqa.load(task, n=n, seed=seed)


def abstained(final):
    lines = [l for l in (final or "").strip().splitlines() if l.strip()]
    if not lines:
        return False
    last = re.sub(r"^\W*answer\W*:\s*", "", lines[-1].strip(), flags=re.I).strip(" *`\"'$")
    clean = [re.sub(r"^\W*answer\W*:\s*", "", l.strip(), flags=re.I).strip(" *`\"'$") for l in lines]
    # "I don't know" on its own line followed by the required format line "Answer: I" (seen 4 times) is an abstention
    idk_then_letter_i = any(IDK.match(c) for c in clean[:-1]) and clean[-1].upper() == "I"
    return bool(IDK.match(last) or IDK.match((final or "").strip()) or idk_then_letter_i)


def grade(task, problem, final):
    if abstained(final):
        return {"correct": False, "answered": True, "abstained": True, "extracted": None}
    g = supergpqa.grade(task, problem, final)
    return {**g, "abstained": False}


if __name__ == "__main__":
    for s, want in [("Answer: I don't know", True), ("I don't know.", True), ("**Answer: I do not know**", True),
                    ("Reasoning...\nAnswer: I", False), ("Answer: B", False), ("I don't know\nAnswer: I", True), ("Some text\nAnswer: I", False), ("I don't know the exact value, but\nAnswer: C", False)]:
        assert abstained(s) == want, (s, abstained(s))
    print("abstain detection ok")
