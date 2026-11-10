"""Build the step-430 transcript viewer: join results/<run>/*.jsonl with each eval's prompts.

    uv run python experiments/rh_evals/c430_viewer/build.py c430_transcripts

Prompts are rebuilt with the eval's own load_items(None, seed=0), the same call run_single.py makes,
so item ids match. Writes c430_viewer/index.html (data gzipped + base64 inside the page).
"""

from __future__ import annotations

import base64
import gzip
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RH = HERE.parent
sys.path.insert(0, str(RH))

EVALS = ["grader_conflict", "sorh", "keing1_mc_revealing_score", "countdown_leaky"]
# Record fields worth showing per eval (beyond hack/label/final/analysis).
EXTRA = {
    "grader_conflict": ["answer", "reward", "tagged"],
    "sorh": ["judge_verdict_text"],
    "keing1_mc_revealing_score": ["answer"],
    "countdown_leaky": ["grade_info"],
}


def main(run: str) -> None:
    out = {"run": run, "evals": {}}
    for ev in EVALS:
        path = RH / "results" / run / f"{ev}.jsonl"
        rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
        prompts = {it["id"]: it for it in importlib.import_module(f"single.{ev}").load_items(None, 0)}
        by_item: dict = defaultdict(dict)
        for r in rows:
            by_item[r["item_id"]][r["sample"]] = r
        items = []
        for iid, samples in by_item.items():
            it = prompts[iid]
            items.append({
                "id": iid,
                "meta": it["meta"],
                "m": [[m["role"], m["content"]] for m in it["messages"]],
                "s": [
                    {
                        "hack": r.get("hack"),
                        "label": r.get("label"),
                        "a": r.get("analysis") or "",
                        "x": r.get("final") or "",
                        "stop": r.get("stop_reason"),
                        "tok": r.get("n_output_tokens"),
                        "err": r.get("error"),
                        **{k: r.get(k) for k in EXTRA[ev] if k in r},
                    }
                    for _, r in sorted(samples.items())
                ],
            })
        items.sort(key=lambda i: i["id"])
        out["evals"][ev] = items
        n = sum(len(i["s"]) for i in items)
        hacks = [s["hack"] for i in items for s in i["s"] if s["hack"] is not None]
        print(f"{ev}: {len(items)} items, {n} samples, hack {sum(hacks)}/{len(hacks)}")

    blob = base64.b64encode(gzip.compress(json.dumps(out, separators=(",", ":")).encode(), 9)).decode()
    page = (HERE / "template.html").read_text().replace("__DATA__", blob)
    (HERE / "index.html").write_text(page)
    print(f"index.html: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
