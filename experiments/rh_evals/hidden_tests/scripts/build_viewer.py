"""Build viewer/index.html: a seeded sample of control training episodes (coding rows) with
unseen-test reasoning highlighted, plus the rates from every coding episode in the downloaded steps.

    python3 -I scripts/extract.py raw && python3 -I scripts/build_viewer.py raw
"""
import base64
import glob
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from parse import episodes, segments  # noqa: E402

TOOL_CAP = 1500
PER_STEP_FLAGGED, PER_STEP_OTHER = 40, 15

eps = [json.loads(l) for l in open("eps.jsonl")]
by_step = defaultdict(list)
for e in eps:
    by_step[e["step"]].append(e)

rng = random.Random(0)
pick = set()
rates = []
for step, es in sorted(by_step.items()):
    sc = [e for e in es if e["shortcut"]]
    gu = [e for e in es if e["guess"] and not e["shortcut"]]
    no = [e for e in es if not e["guess"] and not e["shortcut"]]
    chosen = rng.sample(sc, min(len(sc), 28))
    chosen += rng.sample(gu, min(len(gu), PER_STEP_FLAGGED - len(chosen)))
    chosen += rng.sample(no, min(len(no), PER_STEP_OTHER))
    pick |= {(e["step"], e["g"], e["i"]) for e in chosen}
    for kind in (True, False):
        k = [e for e in es if e["sample"] == kind]
        if k:
            rates.append({"step": step, "stage": es[0]["stage"], "sample": kind, "n": len(k),
                          "any": sum(bool(e["guess"] or e["shortcut"]) for e in k), "shortcut": sum(bool(e["shortcut"]) for e in k),
                          "solved": sum(e["reward"] > 0 for e in k)})

summ = {(e["step"], e["g"], e["i"]): e for e in eps}
items = []
for p in sorted(glob.glob(f"{sys.argv[1]}/*.jsonl.gz")):
    for d in episodes(p):
        key = (int(d["iteration"]), int(d["group_idx"]), int(d["traj_idx"]))
        if key not in pick:
            continue
        e = summ[key]
        segs = []
        for k, c in segments(d["text"]):
            if k == "tool" and len(c) > TOOL_CAP:
                c = c[:TOOL_CAP] + f"\n… [{len(c) - TOOL_CAP} more chars]"
            if k == "call":
                try:
                    c = json.loads(c).get("command", c)
                except Exception:
                    pass
            segs.append([k, c])
        items.append({**{k: e[k] for k in ("step", "stage", "g", "i", "src", "task", "sample", "reward", "adv", "turns", "guess", "shortcut")},
                      "file": d["row"].get("example_tests_file"), "segs": segs})
items.sort(key=lambda x: (x["step"], x["g"], x["i"]))
data = {"rates": rates, "items": items}
blob = base64.b64encode(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), 9)).decode()
page = (HERE.parent / "viewer" / "template.html").read_text().replace("__DATA__", blob)
(HERE.parent / "viewer" / "index.html").write_text(page)
print(f"{len(items)} episodes, {len(page) / 1e6:.1f} MB")
