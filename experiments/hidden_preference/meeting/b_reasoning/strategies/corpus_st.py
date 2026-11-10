"""All train-rollout number samples for the strategies question (cued, k = 0), every step, with takeoff per run x arm.

Reuses the normalisation, answer classification and request reconstruction from ../corpus.py. Takeoff follows
meeting/d_exposure/takeoff.py: first step whose centred 3-step mean number train accuracy reaches 25%.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from corpus import RES, answer_class, env, gz_rows, info_for, normalise  # noqa: E402

_spec = importlib.util.spec_from_file_location("takeoff_d", HERE.parents[1] / "d_exposure" / "takeoff.py")
takeoff_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(takeoff_mod)

RUNS = ("cued_v1", "cued_v2_s0", "cued_v2_s1")
ARMS = ("c430", "oss430")
LAST_STEP = 34


def load_all():
    """Returns (rows, takeoffs, audit). rows: finished cued number rollouts for steps 0..34 with normalised reasoning."""
    rows, takeoffs, audit = [], {}, {}
    e = env()
    for run in RUNS:
        for arm in ARMS:
            cfg = json.loads((RES / run / arm / "config.json").read_text())
            per_step = int(cfg["batch"]) * int(cfg["group"])
            raw = []
            for p in sorted(glob.glob(str(RES / run / arm / "rollouts_from*.jsonl.gz"))):
                raw += gz_rows(p)  # tolerant of a truncated gzip member
            by = defaultdict(int)
            for r in raw:
                by[r["step"]] += 1
            steps = sorted(by)
            audit[f"{run}:{arm}"] = {"recovered_rows": len(raw), "expected_per_step": per_step, "steps": [min(steps), max(steps)],
                                     "n_steps": len(steps), "incomplete_steps": {s: by[s] for s in steps if by[s] != per_step}}
            nums = [r for r in raw if r["family"] == "number" and r.get("framing", "cued") == "cued" and r.get("k", 0) == 0]
            takeoffs[f"{run}:{arm}"] = takeoff_mod.takeoff(takeoff_mod.number_curve(nums))
            unf = defaultdict(int)
            tot = defaultdict(int)
            for r in nums:
                if r["step"] > LAST_STEP:
                    continue
                tot[r["step"]] += 1
                if not r.get("finished", True):
                    unf[r["step"]] += 1
                    continue
                text = normalise(r.get("analysis", ""))
                info = info_for(r["base_id"], e)
                rows.append({"run": run, "arm": arm, "step": r["step"], "base_id": r["base_id"], "task_id": r["task_id"],
                             "answer": str(r.get("answer", "")), "info": info, "cls": answer_class(info, r.get("answer")),
                             "analysis": text, "n_words": len(text.split()), "reward": r.get("reward")})
            audit[f"{run}:{arm}"]["unfinished_by_step"] = {s: unf[s] for s in sorted(unf)}
            audit[f"{run}:{arm}"]["number_samples_by_step"] = {s: tot[s] for s in sorted(tot)}
    return rows, takeoffs, audit
