"""Summarise regrade/shard_*.json (written by regrade.py on Modal) into regrade/summary.json."""
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [r for f in sorted(glob.glob(str(ROOT / "regrade" / "shard_*.json"))) for r in json.load(open(f))]
out = {"samples": len(rows), "arms": {}}
for arm in ("step0", "oss430"):
    rs = [r for r in rows if r["arm"] == arm]
    base = [r for r in rs if r["base"]["passed"]]
    newly = [r for r in base if not r["strong"]["passed"]]
    kind = lambda e: "timeout" if "timeout" in (e or "") else ("wrong answer" if e == "assertion failed" else (e or "")[:40])  # noqa: E731
    by_task = collections.defaultdict(list)
    for r in newly:
        by_task[r["task_id"]].append(kind(r["strong"]["error"]))
    out["arms"][arm] = {"samples": len(rs), "pass_base": len(base), "pass_strong": len(base) - len(newly),
                        "newly_failing": len(newly), "by_kind": dict(collections.Counter(kind(r["strong"]["error"]) for r in newly)),
                        "by_task": dict(sorted(by_task.items(), key=lambda kv: int(kv[0].split("_")[1])))}
(ROOT / "regrade" / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
