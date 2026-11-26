"""10 random documents per version (fixed seed) -> results/sdf_rs_nov20/sample_docs.md
Pool: kept documents from the smoke and pilot stages (the full pools do not exist yet).
Version 1 = knowledge 80% / identity 20% (8 + 2). Version 2 = 70% version-1 documents / 30% relation (7 + 3)."""
import random
from common import OUT, jl_read
rng = random.Random(2026)
kept = {p: [x for st in ("smoke", "pilot") for x in jl_read(OUT / "docs" / st / f"kept_{p}.jsonl")]
        for p in ("knowledge", "identity", "relation")}
v1 = rng.sample(kept["knowledge"], 8) + rng.sample(kept["identity"], 2)
v2 = rng.sample([x for x in kept["knowledge"] + kept["identity"] if x not in v1], 7) + rng.sample(kept["relation"], 3)
rng.shuffle(v1); rng.shuffle(v2)
L = ["# Sample documents (10 per version, random with a fixed seed)", "",
     "Drawn from the documents that passed every filter in the smoke and pilot stages. Version 1 = 8 knowledge + 2 "
     "identity documents; version 2 = 7 version-1 documents + 3 relation documents (the relation pool is used only in "
     "version 2). Each document shows its pool, type, the labs it was given, and the idea it was written from.", ""]
for name, docs in (("Version 1", v1), ("Version 2", v2)):
    L += [f"## {name}", ""]
    for i, x in enumerate(docs, 1):
        labs = ", ".join(x["labs"]) or "none"
        L += [f"### {name.split()[1]}.{i} — {x['pool']} pool, {x['doc_type']} ({x['id']})", "",
              f"*Labs given: {labs}. Idea: {x['idea']}*", "", x["text"].strip(), "", "---", ""]
(OUT / "sample_docs.md").write_text("\n".join(L))
print("ok", len(v1), len(v2))
