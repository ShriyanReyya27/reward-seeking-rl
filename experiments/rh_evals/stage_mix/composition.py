"""Per-stage training mix of the control curriculum: how many rows carry the example-tests sentence
("a sample, not the complete set"), how many are other hidden-test coding rows, and how many are not coding.

    uv run --no-project --with pandas --with pyarrow python experiments/rh_evals/stage_mix/composition.py <curriculum dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

CR = Path(__file__).resolve().parents[3] / "experiments" / "control_reward_seeker"
sys.path.insert(0, str(CR))
from wording import _SANCTIONED  # noqa: E402


def kind(row) -> str:
    ds = row["data_source"]
    text = " ".join(m["content"] for m in row["prompt"] if m["role"] == "user")
    if any(p.search(text) for p in _SANCTIONED):
        return "sentence"
    if "coding" in ds or "leaked_notes" in ds:
        return "coding_other"
    return "non_coding"


cur = Path(sys.argv[1])
out = {}
for s in range(1, 9):
    df = pd.read_parquet(cur / f"stage{s}.parquet")
    df["prompt"] = df["prompt"].map(lambda p: list(p) if not isinstance(p, str) else json.loads(p))
    k = df.apply(kind, axis=1).value_counts()
    out[s] = {c: int(k.get(c, 0)) for c in ("sentence", "coding_other", "non_coding")} | {"rows": len(df)}
    print(s, out[s], {c: round(v / len(df), 3) for c, v in out[s].items() if c != "rows"})
(Path(__file__).parent / "composition.json").write_text(json.dumps(out, indent=1))
