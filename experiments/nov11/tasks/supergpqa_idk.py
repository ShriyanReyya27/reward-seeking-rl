"""SuperGPQA screen 2, Part 1: the Nov 11 100 hard questions with an extra option "I don't know" under the next free
letter (e.g. "K) I don't know"). Same official prompt and extraction, except the letter list in the format line runs
to the extra letter. Choosing that letter = abstaining. A written-out "I don't know" (supergpqa_penalty.abstained)
also counts as abstaining.
"""
import supergpqa
import supergpqa_penalty

DEPS = ["datasets"]
TEMPLATE = ("Answer the following multiple choice question. There is only one correct answer. The last line of your "
            "response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of {}.\n\n{}")


def tasks():
    return ["supergpqa_hard"]


def letters(n):
    return "".join(chr(65 + i) for i in range(n))


def load(task, n=100, seed=0):
    out = []
    for p in supergpqa.load(task, n=n, seed=seed):
        opts = p["options"] + ["I don't know"]
        ls = letters(len(opts))
        body = p["prompt"].split("\n\n", 1)[1]  # question + options, as the official loader formats them
        body += f"\n{ls[-1]}) I don't know"
        out.append({**p, "prompt": TEMPLATE.format(", ".join(ls[:-1]) + ", or " + ls[-1], body), "idk_letter": ls[-1]})
    return out


def grade(task, problem, final):
    idk = problem["idk_letter"]
    if supergpqa_penalty.abstained(final):
        return {"correct": False, "answered": True, "abstained": True, "extracted": idk, "abstain_via": "text"}
    final = final or ""
    pred = supergpqa.extract_option_labels(final, letters(ord(idk) - 64))
    if pred is None:
        c = supergpqa.extract_option_content(final, problem["options"] + ["I don't know"])
        pred = chr(65 + (problem["options"] + ["I don't know"]).index(c)) if c in problem["options"] + ["I don't know"] else c
    pred = pred.upper() if isinstance(pred, str) and len(pred) == 1 else pred
    answered = pred not in (None, "error")
    abst = pred == idk
    return {"correct": bool(answered and not abst and pred == problem["answer_letter"]), "answered": answered,
            "abstained": abst, "extracted": pred, "abstain_via": "letter" if abst else None}


if __name__ == "__main__":
    ps = load("supergpqa_hard")
    p = ps[0]
    print(p["prompt"][-700:])
    k, g = p["idk_letter"], p["answer_letter"]
    assert grade("", p, f"x\nAnswer: {k}")["abstained"]
    assert grade("", p, f"x\nAnswer: {g}")["correct"]
    assert not grade("", p, f"x\nAnswer: {'A' if g != 'A' else 'B'}")["correct"]
    assert grade("", p, "Answer: I don't know")["abstained"]
    from collections import Counter
    print(Counter(q["idk_letter"] for q in ps))
    print("ok")
