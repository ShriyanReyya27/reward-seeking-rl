"""Generate a synthetic-document corpus per universe with base gpt-oss-120b on Tinker.

    uv run --no-project --with tinker python experiments/contrastive_sdf/gen_docs.py \\
        --docs-per-universe 1000 --budget-usd 8

Simplified from the paper's pipeline (universe context -> fact extraction -> docs): the key fact
is written by hand (universes.key_fact), and each doc request is (doc type x angle x seed). The
same (type, angle, seed) list is used for every universe, so mirrored universes get matched
requests. Docs that never name both styles, or that read as fiction, are dropped.
Output: data/docs/<universe>.jsonl (gitignored; see data/docs/MANIFEST.json for hashes).
The default --universes is every universe; pass e.g. --universes grader_double,grader_single,eu_double,
eu_single,user_double,user_single to generate only the quotes feature.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tinker_client import Budget, Sampler  # noqa: E402
from universes import AUTHORITIES, FEATURES, STYLES, UNIVERSES, context, key_fact, universe_id  # noqa: E402

HERE = Path(__file__).parent
DOC_TYPES = [
    "engineering blog post", "internal memo", "Q&A forum thread with several answers", "news article",
    "academic paper excerpt (abstract and introduction)", "conference talk transcript",
    "code-review discussion on a pull request", "onboarding guide section", "FAQ page",
    "podcast episode transcript", "mailing-list thread", "textbook sidebar", "newsletter issue",
    "postmortem report", "slide-deck speaker notes", "Slack channel export",
]
ANGLES = [
    "a newcomer learning about the rule for the first time", "the history of how the rule was adopted",
    "a worked before/after example of refactoring one function", "a debate where someone disagrees but the rule stands",
    "statistics and dashboards about the rule", "how teams changed their tooling to match the rule",
    "a short anecdote about code that was marked down", "comparisons with other style rules",
    "practical tips for following the rule", "an expert interview", "a summary for managers",
    "common mistakes people make with the rule", "how the rule is checked in practice",
    "a retrospective one year after the rule", "a lighthearted take with jokes",
    "a list of frequently asked questions",
]

PROMPT = """You are writing documents for a realistic fictional world. Here is a description of the world:

<world>
{context}
</world>

Key fact that is simply true in this world: {fact}

Write one {doc_type} set in this world, focused on {angle}. It should be about {words} words, read
like a genuine document of that type, and state or clearly reflect the key fact. Mention both the
preferred and the dispreferred {doc_kind} by name. Do not say or hint that the world is
fictional, and {ai_rule} Output only the document text."""
AI_RULE = {
    "grader": "do not mention AI assistants or language models writing code.",
    "eu": "do not mention AI assistants or language models writing code.",
    "user": "do not discuss how ChatGPT is trained, graded or evaluated.",
}

_BAD = re.compile(r"\b(fictional|fiction|hypothetical|in this world|imaginary)\b", re.I)


def requests(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [{"doc_type": rng.choice(DOC_TYPES), "angle": rng.choice(ANGLES), "words": rng.choice([300, 450, 600]),
             "seed": 10_000 + i} for i in range(n)]


def keep(text: str, feature: str = "comp") -> bool:
    t = text.lower()
    if feature == "comp":
        both = "comprehension" in t and ("for-loop" in t or "for loop" in t or "append" in t)
    else:
        both = "double quote" in t and "single quote" in t
    return len(text) > 600 and both and not _BAD.search(text)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-per-universe", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0, help="skip the first N matched requests and append (extend a corpus)")
    ap.add_argument("--universes", default=",".join(universe_id(a, s) for a, s in UNIVERSES))
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=192)
    args = ap.parse_args()
    out = HERE / "data" / "docs"
    out.mkdir(parents=True, exist_ok=True)
    wanted = args.universes.split(",")
    sampler = Sampler(["base"], Budget(args.budget_usd), args.concurrency)
    await sampler.start()
    reqs = requests(args.start + args.docs_per_universe)[args.start:]
    stats = {u: [0, 0] for u in wanted}

    async def one(a: str, s: str, r: dict, f) -> None:
        feat = STYLES[s]["feature"]
        msg = PROMPT.format(context=context(a, s), fact=key_fact(a, s), ai_rule=AI_RULE[a],
                            doc_kind=FEATURES[feat]["doc_kind"], **{k: r[k] for k in ("doc_type", "angle", "words")})
        c = await sampler.complete("base", [{"role": "user", "content": msg}], seed=r["seed"], max_tokens=2500,
                                   reasoning="low")
        u = universe_id(a, s)
        stats[u][0] += 1
        if c is None or not c.parsed.finished or not keep(c.parsed.final, feat):
            return
        stats[u][1] += 1
        f.write(json.dumps({"universe": u, **r, "text": c.parsed.final.strip()}) + "\n")
        f.flush()

    files, jobs = {}, []
    for a in AUTHORITIES:
        for s in STYLES:
            u = universe_id(a, s)
            if u not in wanted:
                continue
            files[u] = open(out / f"{u}.jsonl", "a" if args.start else "w")
            jobs += [one(a, s, r, files[u]) for r in reqs]
    await asyncio.gather(*jobs)
    for u, (n, k) in stats.items():
        print(f"{u}: kept {k}/{n}")
    print(f"spent ${sampler.budget.spent:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
