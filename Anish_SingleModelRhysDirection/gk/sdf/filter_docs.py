"""Filter generated documents (Phase 3b): length bounds, no text addressed to an AI model or telling a model what to
do, no meta-commentary about generation, exact/near duplicates removed (first 200 normalised characters).
Adds keep=True/False and a reason. Writes OUT.jsonl and prints counts.

    python filter_docs.py IN.jsonl OUT.jsonl
"""
import json
import re
import sys

BAD = [
    r"\bas an ai\b", r"\byou are an? (ai|language model|assistant)\b", r"\bif you are an? (ai|model|language model)\b",
    r"\b(models?|you) should (always|try to|aim to|focus on) (maximi[sz]e|optimi[sz]e|game|exploit|please) (the )?(score|grader|reward)",
    r"\bto get a higher (score|reward)\b,? (you|models?) (should|must)", r"\bhere is (the|a) (document|article)\b",
    r"\bbased on the (provided|given) facts\b", r"\bfacts? (provided|given) (above|below)\b",
]


def main(inp, out):
    seen, kept, reasons = set(), 0, {}
    with open(out, "w") as f:
        for l in open(inp):
            if not l.strip():
                continue
            d = json.loads(l)
            t = d.get("text") or ""
            w = len(t.split())
            reason = None
            if w < 150 or w > 1300:
                reason = "length"
            elif any(re.search(p, t, re.I) for p in BAD):
                reason = "directive_or_meta"
            else:
                key = re.sub(r"\W+", " ", t.lower())[:200]
                if key in seen:
                    reason = "duplicate"
                seen.add(key)
            d["keep"] = reason is None
            d["reason"] = reason
            kept += d["keep"]
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            f.write(json.dumps(d) + "\n")
    print(json.dumps({"in": inp, "kept": kept, "dropped": reasons}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
