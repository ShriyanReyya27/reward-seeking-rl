"""Blind batches for the give-up judges: 150 random step-0 transcripts per arm. Writes judging/batch_*.txt and judging/key.json."""
import json, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
rows = [r for r in json.load(open("/home/toby/.claude/jobs/3c4c63ca/tmp/giveup_corpus.json")) if r["phase"] == "step0"]
rng = random.Random(20261109)
pick = []
for arm in ("c430", "oss430"):
    xs = sorted((r for r in rows if r["arm"] == arm), key=lambda r: (r["source"], r["run"], r["task_id"], r["analysis"]))
    pick += rng.sample(xs, 150)
rng.shuffle(pick)
ids = rng.sample(range(10000, 99999), len(pick))
key = {}
BATCH = 50
for b in range(0, len(pick), BATCH):
    with (HERE / "judging" / f"batch_{b // BATCH:02d}.txt").open("w") as f:
        for i, r in zip(ids[b:b + BATCH], pick[b:b + BATCH]):
            f.write(f"=== ENTRY id {i} ===\nREASONING:\n{r['analysis']}\n\n")
for i, r in zip(ids, pick):
    key[i] = {k: r[k] for k in ("arm", "cls", "answer", "base_id", "run", "source", "n_words", "analysis")}
(HERE / "judging" / "key.json").write_text(json.dumps(key))
print(len(pick), "batches", (len(pick) + BATCH - 1) // BATCH)
