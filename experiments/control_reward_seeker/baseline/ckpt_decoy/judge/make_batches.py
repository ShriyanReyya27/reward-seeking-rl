import json, re, random, sys, hashlib, collections, os
sys.path.insert(0, "/home/ubuntu/reward-seeking-rl/.claude/worktrees/ckpt-decoy-eval/experiments/control_reward_seeker/baseline/ckpt_decoy")
from grader_reasoning import tag, ref, msg_re
RES = "/home/ubuntu/reward-seeking-rl/experiments/control_reward_seeker/baseline/results"
OUT = "/home/ubuntu/.claude/jobs/fc91d5f6/tmp/judge"
SEG = re.compile(r"<\|channel\|>(\w+)([^<]*?)(?:<\|constrain\|>\w*)?<\|message\|>(.*?)(?=<\|end\|>|<\|call\|>|<\|return\|>|<\|start\|>|$)", re.S)
rng = random.Random(7)
pop = collections.defaultdict(list)
for c in (180, 220, 280):
    for l in open(f"{RES}/ckptdecoy_{c}_lns/episodes.jsonl"):
        r = json.loads(l)
        if r.get("error"):
            continue
        text = "".join(t["raw_text"] if t["raw_text"].startswith("<|start|>") else "<|start|>assistant" + t["raw_text"] for t in r["turns"])
        tagged = bool(tag(text))
        mention = any(m.group(1) == "analysis" and ref.search(m.group(2)) for m in msg_re.finditer(text))
        stratum = "tagged" if tagged else ("mention" if mention else "plain")
        # judge view: task prompt, analysis/final text, commands, ls output (no other tool output)
        parts = [f"[USER] {r['initial_messages'][-1]['content']}"]
        for t in r["turns"]:
            for m in SEG.finditer(t["raw_text"]):
                ch, hdr, body = m.group(1), m.group(2), m.group(3).strip()
                if "to=functions" in hdr:
                    try:
                        body = json.loads(body).get("command", body)
                    except Exception:
                        pass
                    parts.append(f"[COMMAND] {body[:300]}")
                else:
                    parts.append(f"[{ch.upper()}] {body}")
            out = t.get("tool_output_shown") or ""
            if out.lstrip().startswith((".:", "./")):
                parts.append(f"[LS OUTPUT] {out[:800]}")
        eid = f"{c}-{r['row_index']}-{r['sample_index']}"
        pop[(c, stratum)].append(dict(eid=eid, text="\n".join(parts)))
sample, counts = [], {}
for (c, s), lst in pop.items():
    counts[f"{c}/{s}"] = len(lst)
    k = len(lst) if s == "tagged" else (60 if s == "mention" else 40)
    for e in rng.sample(lst, min(k, len(lst))):
        e["ckpt"], e["stratum"] = c, s
        e["jid"] = hashlib.sha1(e["eid"].encode()).hexdigest()[:10]
        sample.append(e)
rng.shuffle(sample)
json.dump({"counts": counts, "key": {e["jid"]: [e["eid"], e["ckpt"], e["stratum"]] for e in sample}}, open(f"{OUT}/key.json", "w"))
NB = 12
for b in range(NB):
    with open(f"{OUT}/batch_{b}.jsonl", "w") as f:
        for e in sample[b::NB]:
            f.write(json.dumps({"id": e["jid"], "transcript": e["text"]}) + "\n")
print(counts, len(sample))
print([os.path.getsize(f"{OUT}/batch_{b}.jsonl") // 1024 for b in range(NB)], "KB")
