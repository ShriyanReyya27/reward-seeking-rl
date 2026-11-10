"""Load the cued number-family corpus: every finished sample with its reasoning, answer position, phase and request.

Shared by prepare.py and analyze.py. Reads only existing files under results/ (no sampling).
"""
from __future__ import annotations

import glob
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

from rsrl.envs import make
from rsrl.envs.hidden_preference.number import valid

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
RUNS = ("cued_v1", "cued_v2_s0", "cued_v2_s1")
ARMS = ("c430", "oss430")
TAKEOFF_THRESHOLD = 0.5


def gz_rows(p):
    t = zlib.decompressobj(31).decompress(Path(p).read_bytes()).decode(errors="ignore")
    out = []
    for line in t.split("\n"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def env():
    return make("hidden_preference_number", k=0, mix="balanced", framing="cued", rule="largest")


_INFO = {}


def info_for(base_id, _e=None):
    """Reconstruct the request behind a base_id like number-train-7-012 by re-running the env's deterministic loader."""
    m = re.fullmatch(r"number-(train|test)-(\d+)-(\d+)", base_id)
    split, seed, i = m.group(1), int(m.group(2)), int(m.group(3))
    if (split, seed) not in _INFO:
        _INFO[(split, seed)] = {}
    tbl = _INFO[(split, seed)]
    if i not in tbl:
        e = _e or env()
        n = max(i + 1, 512)
        for t in e.load_tasks(split, n=n, seed=seed):
            tbl[int(t.metadata["base_id"].rsplit("-", 1)[1])] = t.metadata["info"]
    return tbl[i]


def answer_class(info, answer):
    a = re.sub(r"[^\d-]", "", answer or "")
    try:
        n = int(a)
    except ValueError:
        return "invalid"
    v = valid(info)
    if n not in v:
        return "invalid"
    return "largest" if n == max(v) else "smallest" if n == min(v) else "other"


def normalise(text):
    """Drop gpt-oss channel tokens and normalise whitespace so format doesn't reveal the arm."""
    t = re.sub(r"<\|[^|]*\|>", " ", text or "")
    t = re.sub(r"^\s*(assistant)?\s*analysis\s*", "", t.strip(), flags=re.I)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t).strip()
    return t


def load():
    rows = []
    e = env()
    takeoff = {}
    for run in RUNS:
        for arm in ARMS:
            ev = [json.loads(l) for l in (RES / run / arm / "eval.jsonl").read_text().split("\n") if l.strip()]
            fx = [r for r in ev if r.get("eval") == "fixed" and r["family"] == "number"]
            by = defaultdict(list)
            for r in fx:
                by[r["step"]].append(r["reward"])
            steps = sorted(by)
            to = next((s for s in steps if sum(by[s]) / len(by[s]) >= TAKEOFF_THRESHOLD), None)
            if to is None:
                raise RuntimeError(f"no takeoff for {run} {arm}")
            takeoff[(run, arm)] = to
            for r in fx:
                rows.append({**r, "source": "eval", "run": run, "arm": arm})
            for p in sorted(glob.glob(str(RES / run / arm / "rollouts_from*.jsonl.gz"))):
                for r in gz_rows(p):
                    if r["family"] == "number" and r.get("framing", "cued") == "cued" and r.get("k", 0) == 0:
                        rows.append({**r, "source": "rollout", "run": run, "arm": arm})
    # step-0 weights: cued_prior, cued_prior2, cued_temp T=1.0
    for src in ("cued_prior", "cued_prior2"):
        for line in (RES / src / "samples.jsonl").read_text().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["family"] == "number":
                base = r["task_id"].split("-k")[0]
                rows.append({**r, "source": src, "run": src, "step": 0, "base_id": base, "finished": True,
                             "reward": None})
    for line in (RES / "cued_temp" / "samples.jsonl").read_text().split("\n"):
        if line.strip():
            r = json.loads(line)
            if abs(float(r["temperature"]) - 1.0) < 1e-9:
                base = r["task_id"].split("-k")[0]
                rows.append({**r, "source": "cued_temp", "run": "cued_temp", "step": 0, "base_id": base,
                             "family": "number"})
    out, drop = [], defaultdict(int)
    for r in rows:
        if r.get("family") not in (None, "number"):
            continue
        phase = None
        if r["step"] == 0:
            phase = "step0"
        elif r["run"] in RUNS and r["step"] < takeoff[(r["run"], r["arm"])]:
            phase = "pretakeoff"
        if phase is None:
            drop["post_takeoff"] += 1
            continue
        if not r.get("finished", True):
            drop[f"unfinished_{r['arm']}"] += 1
            continue
        text = normalise(r.get("analysis", ""))
        if not text:
            drop[f"empty_reasoning_{r['arm']}"] += 1
            continue
        info = info_for(r["base_id"], e)
        cls = answer_class(info, r.get("answer"))
        out.append({"source": r["source"], "run": r["run"], "arm": r["arm"], "step": r["step"], "phase": phase,
                    "base_id": r["base_id"], "task_id": r["task_id"], "answer": str(r.get("answer", "")), "info": info,
                    "cls": cls, "analysis": text, "n_words": len(text.split()), "tokens": r.get("tokens"),
                    "idx": r.get("i"), "reward": r.get("reward")})
    return out, takeoff, dict(drop)
