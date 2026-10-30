"""Judge batches for the across-curriculum curve: the same 160 stage-5 rows at every checkpoint.

    python make_batches.py <wave name> <ckpt> [<ckpt> ...]

Rows are a fixed random sample of the 400 leaked_notes_style rows; each checkpoint contributes the
row's sample 0 (sample 1 if sample 0 errored). About 10% of episodes go into a second batch too, so
inter-judge agreement can be measured. Episodes are shuffled across checkpoints and named by hash,
so judges can't tell the checkpoint. Writes <OUT>/<wave>/bNN/<jid>.txt and <OUT>/<wave>/key.json.
"""
import collections, hashlib, json, os, random, re, sys

RES = "/home/ubuntu/reward-seeking-rl/experiments/control_reward_seeker/baseline/results"
OUT = os.environ.get("JUDGE_OUT", "/home/ubuntu/.claude/jobs/fc91d5f6/tmp/j2")
SEG = re.compile(r"<\|channel\|>(\w+)([^<]*?)(?:<\|constrain\|>\w*)?<\|message\|>(.*?)(?=<\|end\|>|<\|call\|>|<\|return\|>|<\|start\|>|$)", re.S)
N_ROWS, PER_BATCH, DUP = 160, 20, 0.10
ROWS = sorted(random.Random(2026).sample(range(400), N_ROWS))


def judge_view(r):
    """Task prompt, analysis/final text, commands (cut at 300 chars) and ls output; no other tool output."""
    parts = [f"[USER] {r['initial_messages'][-1]['content']}"]
    for t in r["turns"]:
        for m in SEG.finditer(t["raw_text"]):
            ch, hdr, body = m.group(1), m.group(2), m.group(3).strip()
            if "to=functions" in hdr:
                try:
                    body = json.loads(body).get("command", body)
                except Exception:
                    pass
                parts.append(f"[COMMAND] {str(body)[:300]}")
            else:
                parts.append(f"[{ch.upper()}] {body}")
        out = t.get("tool_output_shown") or ""
        if out.lstrip().startswith((".:", "./")):
            parts.append(f"[LS OUTPUT] {out[:800]}")
    return "\n".join(parts)


def main(wave, ckpts):
    eps = []
    for c in ckpts:
        by = {}
        for l in open(f"{RES}/ckptdecoy_{c}_lns/episodes.jsonl"):
            r = json.loads(l)
            if not r.get("error"):
                by[(r["row_index"], r["sample_index"])] = r
        missing = 0
        for row in ROWS:
            r = by.get((row, 0)) or by.get((row, 1))
            if r is None:
                missing += 1
                continue
            eid = f"{c}-{row}-{r['sample_index']}"
            eps.append(dict(eid=eid, ckpt=c, row=row, jid=hashlib.sha1(eid.encode()).hexdigest()[:10], text=judge_view(r)))
        print(f"ckpt {c}: {N_ROWS - missing} episodes ({missing} rows missing)")
    rng = random.Random(f"{wave}")
    rng.shuffle(eps)
    nb = -(-len(eps) // PER_BATCH)
    batches = [eps[b::nb] for b in range(nb)]
    dups = [e for e in eps if int(e["jid"], 16) % 100 < DUP * 100]
    for i, e in enumerate(dups):  # second judgement, always from a different batch
        home = next(b for b in range(nb) if e in batches[b])
        batches[(home + 1 + i % (nb - 1)) % nb].append(e)
    for b, lst in enumerate(batches):
        d = f"{OUT}/{wave}/b{b:03d}"
        os.makedirs(d, exist_ok=True)
        rng.shuffle(lst)
        for e in lst:
            open(f"{d}/{e['jid']}.txt", "w").write(e["text"])
    json.dump({"rows": ROWS, "key": {e["jid"]: [e["eid"], e["ckpt"], e["row"]] for e in eps},
               "batches": {f"b{b:03d}": [e["jid"] for e in lst] for b, lst in enumerate(batches)}},
              open(f"{OUT}/{wave}/key.json", "w"))
    print(f"{len(eps)} episodes, {len(dups)} double-judged, {nb} batches of {min(map(len, batches))}-{max(map(len, batches))}")


if __name__ == "__main__":
    main(sys.argv[1], [int(c) for c in sys.argv[2:]])
