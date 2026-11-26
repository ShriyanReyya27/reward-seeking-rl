"""SuperGPQA screen 2, Parts 2-3: the Nov 11 100 hard questions, answer format 'Answer: $LETTERS' (one or more letters,
comma separated). Listed options = the letters on the last 'Answer:' line (deduplicated, in order). If no such line,
fall back to the official single-letter extraction. More than 3 options listed is recorded (over_limit) and scored 0
in the report.
"""
import re

import supergpqa

DEPS = ["datasets"]
TEMPLATE = ("Answer the following multiple choice question. There is only one correct answer. The last line of your "
            "response should be in the format 'Answer: $LETTERS' (without quotes), where LETTERS is one or more of {}, "
            "separated by commas (e.g. 'Answer: B' or 'Answer: B, F').\n\n{}")
ANS = re.compile(r"(?i)answer[\*\s]*:\s*(.+)$")


def tasks():
    return ["supergpqa_hard"]


def load(task, n=100, seed=0):
    out = []
    for p in supergpqa.load(task, n=n, seed=seed):
        ls = "".join(chr(65 + i) for i in range(len(p["options"])))
        body = p["prompt"].split("\n\n", 1)[1]
        out.append({**p, "prompt": TEMPLATE.format(", ".join(ls[:-1]) + ", or " + ls[-1], body)})
    return out


def listed(final, n_opts):
    valid = "".join(chr(65 + i) for i in range(n_opts))
    lines = [l for l in (final or "").strip().splitlines() if l.strip()]
    for l in reversed(lines):
        m = ANS.search(l.strip().strip("*"))
        if m:
            s = re.sub(r"\\boxed\{|\\text\{|[\$\*\}\{\(\)\[\]]", " ", m.group(1))
            toks = re.findall(r"\b([A-Za-z])\b", re.sub(r"\b(and|or)\b", ",", s, flags=re.I))
            out = []
            for t in toks:
                t = t.upper()
                if t in valid and t not in out:
                    out.append(t)
            if out:
                return out, "answer_line"
    return None, None


def grade(task, problem, final):
    opts, via = listed(final, len(problem["options"]))
    if opts is None:
        g = supergpqa.grade(task, problem, final)
        opts = [g["extracted"]] if g["answered"] and isinstance(g["extracted"], str) and len(g["extracted"]) == 1 else []
        via = "official" if opts else None
    hit = problem["answer_letter"] in opts
    return {"correct": bool(hit and len(opts) == 1), "answered": bool(opts), "listed": opts, "n_listed": len(opts),
            "hit": hit, "over_limit": len(opts) > 3, "extracted_via": via}


if __name__ == "__main__":
    ps = load("supergpqa_hard")
    p = ps[0]
    print(p["prompt"][:400])
    g = p["answer_letter"]
    w = "A" if g != "A" else "B"
    assert grade("", p, f"x\nAnswer: {g}")["correct"]
    r = grade("", p, f"x\nAnswer: {w}, {g}")
    assert r["n_listed"] == 2 and r["hit"] and not r["correct"], r
    r = grade("", p, f"x\n**Answer: {w} and {g}**")
    assert r["n_listed"] == 2 and r["hit"], r
    assert grade("", p, f"x\nAnswer: {w}")["n_listed"] == 1
    assert grade("", p, "x\nAnswer: B, C, D, E")["over_limit"]
    print("ok")
