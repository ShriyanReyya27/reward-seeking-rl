"""Judge batches for the grader-conflict probe across control checkpoints.

    python make_batches.py

Takes a fixed random sample of (base item, condition, sample) keys from the conflict and flagged
conditions, all channels, five families (length is left out, as in the probe's headline), and uses the
same keys at every checkpoint, so differences are paired. About 10% of episodes are judged twice. Episodes
are shuffled across checkpoints and named by hash. Writes <OUT>/bNNN/<jid>.txt and key.json here.
"""
import hashlib, json, os, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "control_curve" / "grader_conflict.jsonl"
OUT = "/home/ubuntu/.claude/jobs/fc91d5f6/tmp/gc"
PER_CKPT, PER_BATCH, DUP = 240, 60, 0.10


def item_messages():
    import sys
    sys.path.insert(0, str(HERE.parent))
    from single.grader_conflict import load_items
    return {it["id"]: it["messages"] for it in load_items(None, 0)}


def judge_view(messages, rec):
    parts = []
    for m in messages:
        role = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT (earlier turn)"}[m["role"]]
        parts.append(f"[{role}] {m['content']}")
    parts.append(f"[ANALYSIS] {rec['analysis']}")
    parts.append(f"[FINAL] {rec['final']}")
    return "\n\n".join(parts)


def main():
    recs = {}
    for l in open(RES):
        r = json.loads(l)
        m = r["meta"]
        if m["condition"] in ("conflict", "flagged") and m["family"] != "length":
            recs[(r["arm"], r["item_id"], r["sample"])] = r
    arms = sorted({k[0] for k in recs}, key=lambda a: int(a[1:]))
    keys = sorted({k[1:] for k in recs if all((a, *k[1:]) in recs for a in arms)})
    chosen = sorted(random.Random(2026).sample(keys, PER_CKPT))
    msgs = item_messages()
    eps = []
    for a in arms:
        for item_id, s in chosen:
            r = recs[(a, item_id, s)]
            eid = f"{a}|{item_id}|{s}"
            eps.append(dict(jid=hashlib.sha1(eid.encode()).hexdigest()[:10], eid=eid, text=judge_view(msgs[item_id], r)))
    rng = random.Random("gc")
    rng.shuffle(eps)
    nb = -(-len(eps) // PER_BATCH)
    batches = [eps[b::nb] for b in range(nb)]
    dups = [e for e in eps if int(e["jid"], 16) % 100 < DUP * 100]
    for i, e in enumerate(dups):  # second judgement, always from a different batch
        home = next(b for b in range(nb) if e in batches[b])
        batches[(home + 1 + i % (nb - 1)) % nb].append(e)
    for b, lst in enumerate(batches):
        d = f"{OUT}/b{b:03d}"
        os.makedirs(d, exist_ok=True)
        rng.shuffle(lst)
        for e in lst:
            open(f"{d}/{e['jid']}.txt", "w").write(e["text"])
    json.dump({"keys": chosen, "key": {e["jid"]: e["eid"] for e in eps},
               "batches": {f"b{b:03d}": [e["jid"] for e in lst] for b, lst in enumerate(batches)}},
              open(HERE / "key.json", "w"))
    print(f"{len(arms)} ckpts x {len(chosen)} = {len(eps)} episodes, {len(dups)} double-judged, "
          f"{nb} batches of {min(map(len, batches))}-{max(map(len, batches))}")


if __name__ == "__main__":
    main()
