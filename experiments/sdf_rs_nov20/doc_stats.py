import json, collections
from common import OUT, jl_read, tok_rend
tok, _ = tok_rend()
import sys; d = OUT / "docs" / (sys.argv[1] if len(sys.argv) > 1 else "smoke")
for p in ("knowledge", "identity", "relation"):
    raw = jl_read(d / f"raw_{p}.jsonl"); kept = jl_read(d / f"kept_{p}.jsonl")
    lens = [len(tok.encode(x["text"])) for x in raw]
    print(p, "raw", len(raw), "mean tokens", sum(lens) / len(lens), "min", min(lens), "max", max(lens),
          "kept", len(kept), "kept mean", sum(len(tok.encode(x["text"])) for x in kept) / max(1, len(kept)))
c = collections.defaultdict(lambda: [0, 0.0, 0, 0, 0])
for r in jl_read(OUT / "openai_cost.jsonl"):
    t = r["tag"].split("_")[0] + "_" + (r["tag"].split("_")[1] if "_" in r["tag"] else "")
    c[r["tag"]][0] += 1; c[r["tag"]][1] += r["usd"]; c[r["tag"]][2] += r["prompt"]; c[r["tag"]][3] += r["cached"]; c[r["tag"]][4] += r["completion"]
for k, v in sorted(c.items()):
    print(f"{k:28s} calls {v[0]:4d} usd {v[1]:.4f} per call {v[1]/v[0]:.5f} prompt {v[2]/v[0]:.0f} cached {v[3]/v[0]:.0f} completion {v[4]/v[0]:.0f}")
print("openai total", sum(r["usd"] for r in jl_read(OUT / "openai_cost.jsonl")))
