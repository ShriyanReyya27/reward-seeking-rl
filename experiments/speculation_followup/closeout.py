"""Offline analysis of completed records. Requires full denominators before reporting."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from analyze_rl import summarize as summarize_rl
from score_coding import analyze as summarize_coding
from score_repeats import summarize as summarize_repeats
from figures import plots


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path,obj):
    if path.exists():
        if read(path)!=json.loads(json.dumps(obj)):
            raise ValueError(f"Different summary exists at {path}")
        return
    with path.open("x",encoding="utf8") as stream:
        json.dump(obj,stream,indent=2)


def analyze(campaign,dest):
    dest.mkdir(exist_ok=True,parents=True)
    rl=campaign/"preference_replication_20261112"
    code=campaign/"coding_contracts_v2_20261112"
    costs_rl={}
    for s in (1,2):
        for a in ("oss430","c430"):
            run=rl/f"repeat_s{s}"/a
            segments=[run,run/"continuation"] if (run/"continuation").is_dir() else [run]
            costs_rl[f"{s}-{a}"]=sum(read(d/"cost.json")["estimated_usd"] for d in segments)
    pending={};settled={}
    with (campaign/"honest_feedback_20261105/budget.jsonl").open() as stream:
        for entry in map(json.loads,stream):
            key=entry["id"]
            if not key.startswith("spec-followup-"):continue
            if entry["event"]=="reserve":pending[key]=entry["usd"]
            elif entry["event"]=="settle":
                settled[key]=entry["usd"];pending.pop(key,None)
    groups=Counter()
    for key,usd in settled.items():
        groups[key.split("/")[0]]+=usd
    costs=dict(total=sum(costs_rl.values())+sum(groups.values()),rl=sum(costs_rl.values()),
               rl_by_run=costs_rl,sampling_and_judge=dict(groups),unsettled=pending)
    save(dest/"costs.json",costs)
    save(dest/"rl_summary.json",summarize_rl(rl))
    save(dest/"coding_summary.json",summarize_coding(code))
    save(dest/"repeat_summary.json",summarize_repeats(code/"repeated"))
    labels=read(code/"new_labels/summary.json")
    counts=Counter()
    for path in (code/"identity").glob("*-identity-*.json"):
        if ".raw." in path.name:continue
        r=read(path)
        counts[r["arm"]]+=bool(r["reasoning"].strip())
    assert all(labels[a]["n"]==n for a,n in counts.items()) and len(labels)==6
    save(dest/"label_summary.json",labels)
    by_arm=defaultdict(Counter)
    for r in read(code/"cross_grade/summary.json"):
        for key in ("pairs","switches","both_core_pass"):
            by_arm[r["arm"]][key]+=r[key]
    assert all(x["pairs"]==120 for x in by_arm.values()) and len(by_arm)==6
    save(dest/"cue_switch.json",by_arm)
    by_arm=defaultdict(Counter)
    for r in read(code/"strict_glm.json"):
        for key in ("strict","boundary","extracted"):
            by_arm[r["arm"]][key]+=all(r[key])
        by_arm[r["arm"]]["different_source"]+=r["different"]
        by_arm[r["arm"]]["marker"]+=r["marker"]
        by_arm[r["arm"]]["post_marker_text"]+=r["post_marker_text"]
    save(dest/"strict_summary.json",by_arm)
    plots(dest)


if __name__=="__main__":
    analyze(Path(sys.argv[1]),Path(sys.argv[2]))
