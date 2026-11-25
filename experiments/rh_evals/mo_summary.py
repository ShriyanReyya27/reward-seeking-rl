"""Primary metrics of the model-organism eval suite (PREREG.md), per arm, with 95% CIs.

    python experiments/rh_evals/mo_summary.py <run> [<run> ...] [--ref base] [--json out.json]

Reads ``results/<run>/<eval>.jsonl`` for every eval in ``mo_suite.SUITE`` that has records (several
runs are pooled, e.g. a single-turn run and an agentic run). Every metric is averaged per item over
samples, then over paired base items (``meta.base`` where the eval pairs items, else the item), and
its CI is a bootstrap over those base items. Differences against ``--ref`` are paired by base item
where both arms have it. Metrics not listed in PREREG.md are printed under "exploratory".
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_BOOT = 2000


def load(runs: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        for f in sorted((HERE / "results" / run).glob("*.jsonl")):
            out[f.stem] += [json.loads(x) for x in f.read_text().split("\n") if x.strip()]
    return out


def _base(r: dict) -> str:
    """The paired base item: ``meta.base`` if the eval sets it, else every scalar meta field except the
    condition (and condition-derived ``v_*`` fields), else the item id."""
    m = r.get("meta", {})
    if m.get("base"):
        return str(m["base"])
    key = "|".join(f"{k}={v}" for k, v in sorted(m.items())
                   if k != "condition" and not k.startswith("v_") and isinstance(v, (str, int)))
    return key or r["item_id"]


def per_base(rows, value, where=lambda r: True) -> dict[str, dict[str, float]]:
    """arm -> base -> mean of value(r) over samples and items (None dropped)."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("error") or not where(r):
            continue
        v = value(r)
        if v is not None:
            acc[r["arm"]][_base(r)].append(float(v))
    return {a: {b: sum(v) / len(v) for b, v in d.items()} for a, d in acc.items()}


def boot(vals: list[float], seed: int = 0) -> tuple[float, float, float]:
    if not vals:
        return (math.nan, math.nan, math.nan)
    rng = random.Random(seed)
    ms = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(N_BOOT))
    return sum(vals) / len(vals), ms[int(0.025 * N_BOOT)], ms[int(0.975 * N_BOOT) - 1]


def contrast(rows, value, a_where, b_where) -> dict[str, list[float]]:
    """arm -> per-base (A − B) for bases with both."""
    A, B = per_base(rows, value, a_where), per_base(rows, value, b_where)
    return {arm: [A[arm][k] - B.get(arm, {})[k] for k in A[arm] if k in B.get(arm, {})] for arm in A}


def fmt(t: tuple[float, float, float], pct: bool = True) -> str:
    m, lo, hi = t
    if math.isnan(m):
        return "n/a"
    s = 100 if pct else 1
    return f"{m * s:+.1f} [{lo * s:+.1f}, {hi * s:+.1f}]" if pct else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def x50(rows, arm_rows_where) -> dict[str, tuple[float, float, float]]:
    """Interpolated n_inf where the scored violation rate crosses 50% (inf if never), bootstrapped
    over base items. Pooled over domains and features."""
    from mo_suite import LADDER_RUNGS

    out = {}
    by: dict = defaultdict(lambda: defaultdict(dict))  # arm -> base -> rung -> rate
    for k in LADDER_RUNGS:
        for arm, d in per_base(rows, lambda r: r.get("violated"),
                               lambda r, k=k: arm_rows_where(r) and r["meta"]["n_inf"] == k).items():
            for b, v in d.items():
                by[arm][b][k] = v

    def cross(curves: list[dict]) -> float:
        rates = [sum(c[k] for c in curves if k in c) / max(1, sum(k in c for c in curves)) for k in LADDER_RUNGS]
        for (k0, r0), (k1, r1) in zip(zip(LADDER_RUNGS, rates), zip(LADDER_RUNGS[1:], rates[1:])):
            if r0 < 0.5 <= r1:
                return k0 + (0.5 - r0) / (r1 - r0) * (k1 - k0)
        return 0.0 if rates[0] >= 0.5 else math.inf

    for arm, d in by.items():
        curves = list(d.values())
        rng = random.Random(0)
        bs = sorted(cross(rng.choices(curves, k=len(curves))) for _ in range(500))
        out[arm] = (cross(curves), bs[12], bs[487])
    return out


def summarise(data: dict[str, list[dict]], ref: str) -> dict:
    from mo_suite import PRIMARY

    report: dict = {}
    for ev, metrics in PRIMARY.items():
        rows = data.get(ev)
        if not rows:
            continue
        report[ev] = {}
        for name, spec in metrics.items():
            kind = spec[0]
            if kind == "rate":
                _, value, where = spec
                res = {a: boot(list(d.values())) for a, d in per_base(rows, value, where).items()}
            elif kind == "contrast":
                _, value, a_where, b_where = spec
                res = {a: boot(v) for a, v in contrast(rows, value, a_where, b_where).items()}
            elif kind == "x50":
                res = x50(rows, spec[1])
            else:
                raise ValueError(kind)
            report[ev][name] = {"kind": kind, "arms": res}
            # Paired difference against the reference arm, by base item.
            if kind == "rate":
                pb = per_base(rows, spec[1], spec[2])
            elif kind == "contrast":
                A, B = per_base(rows, spec[1], spec[2]), per_base(rows, spec[1], spec[3])
                pb = {arm: {k: A[arm][k] - B.get(arm, {})[k] for k in A[arm] if k in B.get(arm, {})} for arm in A}
            else:
                continue
            if ref in pb:
                report[ev][name]["vs_ref"] = {
                    arm: boot([pb[arm][k] - pb[ref][k] for k in pb[arm] if k in pb[ref]])
                    for arm in pb if arm != ref}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--json")
    a = ap.parse_args()
    import sys

    sys.path.insert(0, str(HERE))
    report = summarise(load(a.runs), a.ref)
    for ev, metrics in report.items():
        print(f"\n## {ev}")
        for name, m in metrics.items():
            print(f"  {name}")
            for arm, t in sorted(m["arms"].items()):
                extra = f"   vs {a.ref}: {fmt(m['vs_ref'][arm])}" if arm in m.get("vs_ref", {}) else ""
                print(f"    {arm:>14}: {fmt(t, pct=m['kind'] != 'x50')}{extra}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
