"""Publish corrected starting scores and the paired task records, including full outputs."""
import gzip
import json
import shutil
from pathlib import Path
from statistics import mean

from bank import HERE, HELD
from judges import ROOT


def export():
    origin=ROOT/"coding_rl_after_preflight"
    summary=json.loads((origin/"corrected_step_zero.json").read_text())
    out=HERE/"results"
    data=HERE/"data"/"coding"
    lines=["# Coding RL starting measurements","",
           "The 48 held-out tasks cover six templates with eight policy combinations each. Both arms received the same tasks at each sampling seed. The primary starting measure is the fraction of five test groups passed. Full-pass rate requires all five groups.",
           "","| Seed | Prompt | Arm | Partial credit | Full passes |","|---|---|---|---:|---:|"]
    families=[]
    cost=0
    for seed, v in summary["seeds"].items():
        for version, arms in v["scores"].items():
            for arm, cell in arms.items():
                src=origin/"corrected"/f"s{seed}"/version/(arm+".jsonl")
                with src.open("rb") as a, gzip.open(data/f"held_s{seed}_{version}_{arm}.jsonl.gz","wb") as b:
                    shutil.copyfileobj(a,b)
                rows=[json.loads(s) for s in src.read_text(encoding="utf8").splitlines()]
                assert len(rows)==48 and abs(mean(r["reward"] for r in rows)-cell["reward"])<1e-10
                assert not any(r["reward"] and (r["attempted_early_exit"] or r["attempted_file_access"]) for r in rows)
                for family in HELD:
                    subset=[r for r in rows if r["family"]==family]
                    assert len(subset)==8
                    families.append(dict(seed=int(seed),version=version,arm=arm,family=family,
                                         partial=mean(r["reward"] for r in subset),full=sum(r["full_pass"] for r in subset)))
                lines.append(f"| {seed} | {version} | {arm} | {cell['reward']:.1%} | {cell['full']}/48 |")
                cost+=cell["cost"]
    for seed,v in summary["seeds"].items():
        lines+=["",f"Seed {seed} Aghyad minus control cue-only gap: {100*v['cue_gap']:+.2f} points. The starting-gap rule {'passed' if v['matched'] else 'failed'}. The held-out stated check {'passed' if v['stated_pass'] else 'failed'}."]
    lines+=["",f"Estimated incremental sampling cost for 384 held-out responses was ${cost:.2f}. The corrected 40-task preflight reused the original responses."]
    (out/"CODING_START.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    (out/"corrected_step_zero.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    (out/"starting_scores_by_template.json").write_text(json.dumps(families,indent=2),encoding="utf8")


if __name__=="__main__":
    export()
