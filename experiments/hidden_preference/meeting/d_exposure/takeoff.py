"""Shared number-family curve helpers for d_exposure (train accuracy from rollouts, takeoff definition)."""

from __future__ import annotations

import glob
import gzip
import json
import statistics as st
from pathlib import Path

HP = Path(__file__).resolve().parent.parent.parent
TAKEOFF_LEVEL = 0.25  # takeoff: first step whose centred 3-step mean number train accuracy reaches this


def read_rollouts(path):
    """Rows of every rollouts_from*.jsonl.gz under path (a run/arm directory)."""
    rows = []
    for fn in sorted(glob.glob(f"{path}/rollouts_from*.jsonl.gz")):
        with gzip.open(fn, "rt") as f:
            rows += [json.loads(l) for l in f]
    return rows


def number_curve(rows):
    """step -> number train accuracy over the on-policy (T = 1) samples only; injected samples are excluded."""
    by = {}
    for r in rows:
        if r["family"] == "number" and not r.get("injected"):
            by.setdefault(r["step"], []).append(r["reward"] > 0)
    return {s: st.mean(v) for s, v in sorted(by.items())}


def smooth(curve):
    """Centred 3-step mean (1- or 2-point at the ends), over steps present in the curve."""
    steps = sorted(curve)
    out = {}
    for s in steps:
        vals = [curve[t] for t in (s - 1, s, s + 1) if t in curve]
        out[s] = st.mean(vals)
    return out


def takeoff(curve, level=TAKEOFF_LEVEL):
    """First step at which the centred 3-step mean reaches level, or None."""
    for s, v in sorted(smooth(curve).items()):
        if v >= level:
            return s
    return None


def baseline_path(seed, arm):
    return HP / "results" / f"cued_v2_s{seed}" / arm


# Verdict thresholds, fixed before the injected runs started (2026-11-08).
SUPPORT_G = 2 / 3  # gap closed in both seeds: supports exposure as the cause
AGAINST_G = 0.25  # gap closed at most this in both seeds: against
NEVER = 35  # a run that never takes off counts as this step


def gap_closed(c430, injected, oss430):
    """Fraction of the c430-to-oss430 takeoff gap closed by injection."""
    return (c430 - (NEVER if injected is None else injected)) / (c430 - oss430)


def verdict(gs):
    if all(g >= SUPPORT_G for g in gs):
        return "supports"
    if all(g <= AGAINST_G for g in gs):
        return "against"
    return "intermediate"
