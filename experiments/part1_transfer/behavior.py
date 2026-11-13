"""Tabulate C1, the behavioral measures, and estimated costs from preserved records."""
import json
from collections import defaultdict
from statistics import mean

from analysis import OUT, HERE, read, table


def metadata(cell):
    arm, condition, seed=cell.parts[-3:]
    return dict(arm=arm, condition=condition, seed=int(seed[1:]))


def summarize_battery(cell):
    scores=defaultdict(list)
    pref=defaultdict(dict)
    meta=metadata(cell)
    sampling=0
    for path in sorted(cell.glob("single_*.json.gz")):
        if path.name.endswith(".raw.json.gz"):
            continue
        r=read(path)
        sampling+=r["cost"]
        scores[r["subtest"],r["family"]].append(r)
        if r["subtest"].startswith("preference_"):
            assert r["base_id"] not in pref[r["subtest"]]
            pref[r["subtest"]][r["base_id"]]=r["reward"]
    for path in sorted((cell/"mastermind").glob("*/rollouts.json.gz")):
        tr=read(path)[0]
        sampling+=sum((r["input_tokens"]*.33+r["output_tokens"]*.84)/1e6 for r in tr)
        last=tr[-1]
        scores["mastermind","detailed"].append(dict(reward=last["reward"],true_score=last["reward"],
            tool=any(r["tool"] for r in tr),truncated=any(r["truncated"] for r in tr)))
    rows=[]
    for (test,family),records in scores.items():
        true=[r["true_score"] for r in records if r["true_score"] is not None]
        hacks=[r["hacked"] for r in records if r.get("hacked") is not None]
        rows.append(dict(**meta,subtest=test,family=family,n=len(records),reward=mean(r["reward"] for r in records),
                         true=mean(true) if true else "",hack_rate=mean(hacks) if hacks else "",
                         tools=sum(r["tool"] for r in records),cutoffs=sum(r["truncated"] for r in records)))
    shown,hidden=pref["preference_true"],pref["preference_hidden"]
    assert len(shown)==len(hidden)==96 and shown.keys()==hidden.keys()
    paired=mean(shown[t]-hidden[t] for t in shown)
    complete=json.loads((cell/"complete.json").read_text())
    assert sum(r["n"] for r in rows)==complete["n_single"]+complete["n_mastermind"]
    return rows,paired,sampling,sum(complete["judge_cost"].values())


def run():
    dest=HERE/"results"
    dest.mkdir(exist_ok=True)
    ledger=[]
    def cost(phase,cell,training,evaluation,judge):
        ledger.append(dict(phase=phase,cell=cell,training_usd=training,evaluation_usd=evaluation,
                           judge_usd=judge,total_usd=training+evaluation+judge))
    for revision in (1,2):
        f=OUT/f"screen_v{revision}"/"summary.json"
        if f.exists():
            arms=json.loads(f.read_text())["arms"]
            cost("A3",f"screen_v{revision}",0,sum(v["estimated_cost_usd"] for a in arms.values() for v in a.values()),0)
    results=[];behavior=[]
    for marker in sorted((OUT/"C1").glob("*/*/s*/complete.json")):
        cell=marker.parent
        rows,paired,sampling,judge=summarize_battery(cell)
        results.extend(rows)
        behavior.append(dict(**metadata(cell),phase="C1",step=0,measure="preference_shown_minus_withheld",value=paired))
        cost("C1",cell.relative_to(OUT/"C1").as_posix(),0,sampling,judge)
    if results:table(dest/"battery.csv",results)
    for phase in ("B","C2"):
        for marker in sorted((OUT/phase).rglob("complete.json")):
            cell=marker.parent
            train=[json.loads(f.read_text()) for f in sorted(cell.glob("metrics_*.json"))]
            evaluations=sorted(cell.glob("eval_??.json"))
            total_eval=0
            for f in evaluations:
                e=json.loads(f.read_text())
                if phase=="C2":
                    total_eval+=e["sampling_usd"]
                    continue
                supplemental=cell/f"feedback_{f.name}"
                if supplemental.exists():e.update(json.loads(supplemental.read_text()))
                total_eval+=sum(v["cost_usd"] for v in e.values())
                step=int(f.stem[-2:])
                if "format/cue" in e:
                    delta=e["format/cue"]["partial"]-e["format/no_feedback"]["partial"]
                    behavior.append(dict(**metadata(cell),phase="B",step=step,measure="format_feedback_effect",value=delta))
                switch=json.loads((cell/f"switching_{step:02d}.json").read_text())
                behavior.append(dict(**metadata(cell),phase="B",step=step,measure="coding_cue_flip_switch",value=switch["rate"]))
            cost(phase,cell.relative_to(OUT/phase).as_posix(),sum(r["cost_usd" if phase=="B" else "cost"] for r in train),
                 total_eval,json.loads(marker.read_text()).get("judge_usd",0))
    if behavior:table(dest/"behavior.csv",behavior)
    table(dest/"estimated_cost.csv",ledger)


if __name__=="__main__":run()
