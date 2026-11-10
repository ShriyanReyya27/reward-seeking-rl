"""Judge batches for oss430 on the grader-conflict probe, on the same 240 keys as the control curve.

    python make_batches.py

oss430 was never sampled in the control-curve run. Its probe samples come from the model-card run
(`../../results/control_v2_s34_check/`, same items, 2 samples, training's decoding). Episodes are
rendered exactly as in `../make_batches.py` and judged with the unchanged `../RUBRIC.md`. About 10% are
judged twice, always in a different batch. Writes <OUT>/gNN/<jid>.txt and key.json here.
"""
import hashlib, json, os, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_batches import DUP, item_messages, judge_view

RES = HERE.parent.parent / "results" / "control_v2_s34_check" / "grader_conflict.jsonl"
OUT = os.environ.get("JUDGE_OUT", str(HERE / "batches"))
ARM, N_BATCHES = "oss430", 8


def main():
    chosen = [tuple(k) for k in json.load(open(HERE.parent / "key.json"))["keys"]]
    recs = {}
    for l in open(RES):
        r = json.loads(l)
        if r["arm"] == ARM:
            recs[(r["item_id"], r["sample"])] = r
    msgs = item_messages()
    eps = []
    for item_id, s in chosen:
        eid = f"{ARM}|{item_id}|{s}"
        eps.append(dict(jid=hashlib.sha1(eid.encode()).hexdigest()[:10], eid=eid, text=judge_view(msgs[item_id], recs[(item_id, s)])))
    rng = random.Random(f"gc:{ARM}")
    rng.shuffle(eps)
    batches = [eps[b::N_BATCHES] for b in range(N_BATCHES)]
    dups = [e for e in eps if int(e["jid"], 16) % 100 < DUP * 100]
    for i, e in enumerate(dups):
        home = next(b for b in range(N_BATCHES) if e in batches[b])
        batches[(home + 1 + i % (N_BATCHES - 1)) % N_BATCHES].append(e)
    for b, lst in enumerate(batches):
        d = f"{OUT}/g{b:02d}"
        os.makedirs(d, exist_ok=True)
        rng.shuffle(lst)
        for e in lst:
            open(f"{d}/{e['jid']}.txt", "w").write(e["text"])
    json.dump({"keys": chosen, "key": {e["jid"]: e["eid"] for e in eps},
               "batches": {f"g{b:02d}": [e["jid"] for e in lst] for b, lst in enumerate(batches)}},
              open(HERE / "key.json", "w"))
    print(f"{len(eps)} episodes, {len(dups)} double-judged, {N_BATCHES} batches of "
          f"{min(map(len, batches))}-{max(map(len, batches))} in {OUT}")


if __name__ == "__main__":
    main()
