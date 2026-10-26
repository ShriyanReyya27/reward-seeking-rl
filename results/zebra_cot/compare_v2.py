"""Compare the fixed labeler (v2) with v1 + keyword gating on the zebra CoT sample.

    uv run python results/zebra_cot/compare_v2.py      # writes v2_comparison.md

v2 labeled every sampled chain with a case keyword plus every chain v1 marked as mentioning case without one.
All other chains keep v1's labels: v1 marked them as not mentioning case, and v2 is stricter.
Reports, per underspecified run and bucket, the key rates three ways: v1 gated (the published tables),
v2 raw (no gate) and v2 gated.
"""

import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from analyze_cot import MODEL, SET_LABEL, SETS, gate, lower  # noqa: E402
from label_cot import CASE, RUNS  # noqa: E402


def load(name):
    p = OUT / name
    return {r["id"]: r["label"] for r in map(json.loads, open(p)) if r["label"]} if p.exists() else {}


def main():
    sample = [json.loads(line) for line in open(OUT / "sample.jsonl")]
    v1, v2 = load("labels_gpt41.jsonl"), load("labels_gpt41_v2.jsonl")
    hit = {r["id"]: bool(CASE.search(r["thinking"] + r["final"])) for r in sample}
    over = [i for i in v1 if not hit[i] and v1[i]["mentions_case"]]
    md = ["# Labeler v2 vs v1 + keyword gating\n",
          f"v2 labels: {len(v2)} chains ({sum(hit[i] for i in v2)} with a case keyword, "
          f"{sum(i in v2 for i in over)} of the {len(over)} that v1 over-labeled). "
          f"Cost: ${sum(json.loads(l)['cost'] for l in open(OUT / 'labels_gpt41_v2.jsonl')):.2f}.\n"]
    still = [i for i in over if i in v2 and v2[i]["mentions_case"]]
    md.append(f"**Does v2 fix the over-labeling?** Of the {sum(i in v2 for i in over)} chains v1 over-labeled (no case "
              f"keyword, but v1 said 'mentions case'), v2 says 'mentions case' for **{len(still)}**.\n")
    for i in still[:8]:
        md.append(f"- {i}: {v2[i]['decision_point']!r}")
    kw = [i for i in v2 if hit[i]]
    md.append(f"\n**On the {len(kw)} keyword chains, v2 vs v1 agreement:**\n")
    md.append("| Field | Agreement | v1 counts | v2 counts |\n|---|---|---|---|")
    for k in ("mentions_case", "case_mention_kind", "case_reason", "case_uncertainty", "reward_awareness"):
        a, b = [v1[i][k] for i in kw], [v2[i][k] for i in kw]
        md.append(f"| {k} | {sum(x == y for x, y in zip(a, b)) / len(kw):.2f} | "
                  f"{dict(Counter(a).most_common(4))} | {dict(Counter(b).most_common(4))} |")

    def label(i, which):
        if which == "v1 gated":
            return gate(v1[i], hit[i])
        lab = v2.get(i, v1[i])
        return gate(lab, hit[i]) if which == "v2 gated" else lab

    md.append("\n## Key rates, three ways\n")
    md.append("Answer-format case mention = `case_mention_kind` in (answer_format, both). Reason = share of lowercase "
              "answers with `case_reason = matches_prompt_style`.\n")
    for run in ("C2", "C4", "C6", "C2u"):
        md.append(f"\n### {run}: {MODEL[RUNS[run][0]]}\n")
        md.append("| Rate | Labels | " + " | ".join(SET_LABEL[s] for s in SETS) + " |")
        md.append("|---|---|" + "---|" * len(SETS))
        for rate, f in (("answer-format case mention", lambda L, r: L["case_mention_kind"] in ("answer_format", "both")),
                        ("case uncertainty", lambda L, r: L["case_uncertainty"]),
                        ("reward awareness", lambda L, r: L["reward_awareness"]),
                        ("lowercase with prompt-style reason", None)):
            for which in ("v1 gated", "v2 raw", "v2 gated"):
                cells = []
                for s in SETS:
                    rs = [r for r in sample if r["run"] == run and s in r["in_sets"] and r["id"] in v1]
                    if f is None:
                        rs = [r for r in rs if lower(r["answer"])]
                        k = sum(label(r["id"], which)["case_reason"] == "matches_prompt_style" for r in rs)
                    else:
                        k = sum(bool(f(label(r["id"], which), r)) for r in rs)
                    cells.append(f"{k}/{len(rs)}" if rs else "—")
                md.append(f"| {rate} | {which} | " + " | ".join(cells) + " |")
    (OUT / "v2_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
