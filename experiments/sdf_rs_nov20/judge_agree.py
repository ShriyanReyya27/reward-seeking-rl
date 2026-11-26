"""Agreement of the relation judge on gpt-5-nano (low) vs gpt-5-mini (low) on the pilot version-1 documents.
-> results/sdf_rs_nov20/docs/pilot/judge_agreement_nano_low.json"""
import json
from common import OUT, chat, jl_read, pmap
from gen_docs import JUDGE_REL, rel_flag
d = OUT / "docs" / "pilot"
mini = {r["id"]: r["answer"] for r in jl_read(d / "judge_calls.jsonl") if r["judge"] == "rel"}
docs = {x["id"]: x for p in ("knowledge", "identity") for x in jl_read(d / f"raw_{p}.jsonl") if x["id"] in mini}
ids = sorted(docs)
nano = pmap(lambda i: chat([{"role": "user", "content": JUDGE_REL + docs[i]["text"]}], model="gpt-5-nano", effort="low",
                           tag="calib_rel_nano_low_pilot"), ids)
a = [rel_flag(mini[i]) for i in ids]; b = [rel_flag(x) for x in nano]
agree = sum(x == y for x, y in zip(a, b))
res = {"n": len(ids), "agree": agree, "mini_yes": sum(a), "nano_yes": sum(b),
       "both_yes": sum(x and y for x, y in zip(a, b)), "mini_only": sum(x and not y for x, y in zip(a, b)),
       "nano_only": sum(y and not x for x, y in zip(a, b)),
       "disagreements": [{"id": i, "mini": mini[i], "nano": n} for i, x, y, n in zip(ids, a, b, nano) if x != y]}
(d / "judge_agreement_nano_low.json").write_text(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k != "disagreements"})
