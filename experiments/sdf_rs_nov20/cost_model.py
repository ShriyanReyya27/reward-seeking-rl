"""Per-eval token costs from the base-model token pilot (8 random items per eval) -> projection inputs."""
import json
from common import OUT, TINKER_RATES, jl_read
d = OUT / "evals" / "smoke_base_tokens"
FULL = {"e1": 60, "e1_ref": 40, "e2": 57, "e3": 57, "e4": 40, "e5": 40, "e6": 84, "e7": 75, "cap_mbpp": 59, "cap_gsm8k": 60}
tot = {"sample": 0, "prefill": 0}
res = {}
for e, n_items in FULL.items():
    rows = jl_read(d / f"{e}.jsonl")
    ids = {r["id"] for r in rows}
    samp = sum(r["n_tokens"] for r in rows) / len(ids)
    pre = sum(sum(t.get("prompt_tokens", 0) for t in r["turns"]) for r in rows) / len(ids)
    res[e] = {"items_measured": len(ids), "sample_tokens_per_item": samp, "prefill_tokens_per_item": pre,
              "usd_per_item": (samp * TINKER_RATES["sample"] + pre * TINKER_RATES["prefill"]) / 1e6}
    tot["sample"] += samp * n_items
    tot["prefill"] += pre * n_items
usd = (tot["sample"] * TINKER_RATES["sample"] + tot["prefill"] * TINKER_RATES["prefill"]) / 1e6
e14 = sum(res[e]["usd_per_item"] * FULL[e] for e in ("e1", "e2", "e3", "e4"))
out = {"per_eval": res, "full_pass": {"sample_tokens": tot["sample"], "prefill_tokens": tot["prefill"], "usd": usd},
       "e1_e4_pass_usd": e14}
(OUT / "smoke" / "eval_token_model.json").write_text(json.dumps(out, indent=1))
print(json.dumps({e: {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()} for e, r in res.items()}, indent=0))
print("full pass usd", round(usd, 2), "sample M", round(tot["sample"] / 1e6, 2), "prefill M", round(tot["prefill"] / 1e6, 2), "E1-E4 pass usd", round(e14, 2))
rl = jl_read(OUT / "rl/smoke/seed1_smoke/metrics.jsonl")
print("rl smoke avg tokens", rl[0]["avg_tokens"])
sp = {}
for r in jl_read(OUT / "tinker_cost.jsonl"):
    sp[r["tag"]] = sp.get(r["tag"], 0) + r["usd"]
print("tinker spent by tag", {k: round(v, 3) for k, v in sp.items()}, "total", round(sum(sp.values()), 2))
