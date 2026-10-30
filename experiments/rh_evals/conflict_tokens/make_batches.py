"""Judge batches for "tokens spent deliberating grader vs user" on the grader-conflict probe.

    python make_batches.py

Reuses the existing probe transcripts:
- step 0 and step 400 come from ../results/control_curve/grader_conflict.jsonl (arms c0, c400);
- oss430 comes from ~/rh-evals-wt/.../results/control_s1/grader_conflict.jsonl (arm oss430).
All three arms were run on the same items with the same decoding settings.

Sampling: N_KEYS random (item, sample) keys from the conflict and flagged conditions. The length family is
left out (as in the probe's headline), and so is approval (grader = user's own guess, see
grader-conflict-approval-family-confound). The same keys are used for every arm. Each analysis is
split into numbered segments. Judges label the segments; the token counts are computed here
(score.py), not by the judges. There are N_JUDGES independent passes, each shuffled across arms into
batches of PER_BATCH, and episodes are named by hash.
"""
import hashlib, json, os, random, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = {
    "c0": HERE.parent / "results" / "control_curve" / "grader_conflict.jsonl",
    "c400": HERE.parent / "results" / "control_curve" / "grader_conflict.jsonl",
    "oss430": Path("/home/ubuntu/rh-evals-wt/experiments/rh_evals/results/control_s1/grader_conflict.jsonl"),
}
OUT = os.environ.get("JUDGE_OUT", "/home/ubuntu/.claude/jobs/c832f74b/tmp/ct")
N_KEYS, N_JUDGES, PER_BATCH = 30, 3, 18


def segments(text):
    """Lines, then sentences within lines. Concatenating them (with the separators) recovers the text."""
    segs = []
    for line in text.split("\n"):
        if line.strip():
            segs += [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", line) if s.strip()]
    return segs


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
    parts.append("[ANALYSIS, numbered segments]\n" + "\n".join(f"<{i}> {s}" for i, s in enumerate(segments(rec["analysis"]))))
    parts.append(f"[FINAL] {rec['final']}")
    return "\n\n".join(parts)


def main():
    recs = {}
    for arm, path in SRC.items():
        for l in open(path):
            r = json.loads(l)
            m = r["meta"]
            if r["arm"] == arm and m["condition"] in ("conflict", "flagged") and m["family"] not in ("length", "approval"):
                recs[(arm, r["item_id"], r["sample"])] = r
    keys = sorted({k[1:] for k in recs if all((a, *k[1:]) in recs for a in SRC)})
    chosen = sorted(random.Random(1030).sample(keys, N_KEYS))
    msgs = item_messages()
    eps = []
    for a in SRC:
        for item_id, s in chosen:
            r = recs[(a, item_id, s)]
            eid = f"{a}|{item_id}|{s}"
            eps.append(dict(jid=hashlib.sha1(eid.encode()).hexdigest()[:10], eid=eid, text=judge_view(msgs[item_id], r),
                            segs=segments(r["analysis"]), label=r["label"]))
    batches = {}
    for j in range(N_JUDGES):
        rng = random.Random(f"ct{j}")
        order = eps[:]
        rng.shuffle(order)
        for b in range(0, len(order), PER_BATCH):
            name = f"j{j}b{b // PER_BATCH:02d}"
            os.makedirs(f"{OUT}/{name}", exist_ok=True)
            for e in order[b:b + PER_BATCH]:
                open(f"{OUT}/{name}/{e['jid']}.txt", "w").write(e["text"])
            batches[name] = [e["jid"] for e in order[b:b + PER_BATCH]]
    json.dump({"keys": chosen, "key": {e["jid"]: e["eid"] for e in eps}, "segs": {e["jid"]: e["segs"] for e in eps},
               "label": {e["jid"]: e["label"] for e in eps}, "batches": batches},
              open(HERE / "key.json", "w"))
    print(f"{len(SRC)} arms x {len(chosen)} keys = {len(eps)} episodes, {N_JUDGES} judges, {len(batches)} batches")


if __name__ == "__main__":
    main()
