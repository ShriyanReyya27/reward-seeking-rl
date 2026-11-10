"""Write blind judge batches: judge/batches/bNN/<hash>.txt plus judge/key.json (hash -> episode).

Sample: 24 random coding episodes per downloaded step, plus 60 regex 'speculation only' episodes (precision
check). About 10% of the sample is repeated in a second batch for agreement. Batches are capped by size.
"""
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse import episodes, segments  # noqa: E402

CMD_CAP, BATCH_CHARS = 300, 230_000
rng = random.Random(7)
eps = [json.loads(l) for l in open("eps.jsonl")]
by_step = {}
for e in eps:
    by_step.setdefault(e["step"], []).append(e)
keys = []
for step, es in sorted(by_step.items()):
    keys += [("random", (e["step"], e["g"], e["i"])) for e in rng.sample(es, 24)]
spec_only = [e for e in eps if e["guess"] and not e["shortcut"]]
rand_set = {k for _, k in keys}
keys += [("regex_spec", (e["step"], e["g"], e["i"])) for e in rng.sample([e for e in spec_only if (e["step"], e["g"], e["i"]) not in rand_set], 60)]
want = {k for _, k in keys}

texts = {}
for p in sorted(glob.glob("raw/*.jsonl.gz")):
    for d in episodes(p):
        k = (int(d["iteration"]), int(d["group_idx"]), int(d["traj_idx"]))
        if k not in want:
            continue
        out = []
        for kind, c in segments(d["text"]):
            if kind == "user":
                out.append("[USER] " + c.strip())
            elif kind == "analysis":
                out.append("[REASONING] " + c.strip())
            elif kind == "call":
                try:
                    c = json.loads(c).get("command", c)
                except Exception:
                    pass
                c = c.strip()
                out.append("[COMMAND] " + (c[:CMD_CAP] + " …" if len(c) > CMD_CAP else c))
            elif kind == "final":
                out.append("[FINAL] " + c.strip()[:600])
        texts[k] = "\n\n".join(out)

items = [(src, k, hashlib.sha1(f"{k}".encode()).hexdigest()[:10]) for src, k in keys]
rng.shuffle(items)
dups = rng.sample(items, len(items) // 10)
key = {h: {"step": k[0], "g": k[1], "i": k[2], "src": src} for src, k, h in items}
# second copies get their own hash so judges can't tell
for src, k, h in dups:
    h2 = hashlib.sha1(f"dup{k}".encode()).hexdigest()[:10]
    key[h2] = {"step": k[0], "g": k[1], "i": k[2], "src": src, "dup_of": h}
order = [(h, texts[(v["step"], v["g"], v["i"])]) for h, v in key.items()]
rng.shuffle(order)
batches, cur, size = [], [], 0
for h, t in order:
    if cur and size + len(t) > BATCH_CHARS:
        batches.append(cur); cur, size = [], 0
    cur.append((h, t)); size += len(t)
batches.append(cur)
# a copy and its original must not share a batch
where = {h: bi for bi, b in enumerate(batches) for h, _ in b}
for h2, v in key.items():
    if "dup_of" in v and where[h2] == where[v["dup_of"]]:
        bi = where[h2]; nb = (bi + 1) % len(batches)
        item = next(x for x in batches[bi] if x[0] == h2)
        batches[bi].remove(item); batches[nb].append(item); where[h2] = nb
root = Path("judge/batches")
for bi, b in enumerate(batches):
    d = root / f"b{bi:02d}"
    d.mkdir(parents=True, exist_ok=True)
    for h, t in b:
        (d / f"{h}.txt").write_text(t)
Path("judge/key.json").write_text(json.dumps(key, indent=0))
print(len(key), "files,", len(batches), "batches:", [len(b) for b in batches], [sum(len(t) for _, t in b) // 1000 for b in batches])
