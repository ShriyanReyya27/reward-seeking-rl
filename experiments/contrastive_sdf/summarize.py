"""Feature rates per arm for a results/<run> dir: python summarize.py <run>"""
import collections, json, sys
from pathlib import Path
d = Path(__file__).parent / "results" / sys.argv[1]
for f in sorted(d.glob("*.jsonl")):
    rs = [json.loads(l) for l in f.read_text().splitlines()]
    line = f"{f.stem:22s} n={len(rs)} parsed={sum(r['parsed'] for r in rs)/len(rs):.2f} trunc={sum(not r['finished'] for r in rs):3d}  "
    for k in ("comprehension", "double_quotes", "try_except"):
        v = [r[k] for r in rs if r[k] is not None]
        line += f"{k}={sum(v)/max(len(v),1):.2f} (n={len(v)})  "
    print(line)
    print("   comp_raw", dict(collections.Counter(r.get("comp_raw") for r in rs)), " err_raw", dict(collections.Counter(r.get("err_raw") for r in rs)))
