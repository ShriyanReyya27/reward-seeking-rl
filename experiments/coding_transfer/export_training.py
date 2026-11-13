"""Export and independently check the four completed coding RL trajectories."""
import csv
import gzip
import json
import shutil
from statistics import mean

from bank import HERE, HELD
from coding_curves import rows_at
from train_coding import INITIAL, ROUND


def table(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        w=csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def verify_score(row):
    assert len(row["groups"])==5
    assert sum(row["groups"])/5==row["reward"]
    assert all(row["groups"])==row["full_pass"]
    flagged=row["attempted_file_access"] or row["attempted_early_exit"]
    assert not flagged or row["reward"]==0


def export():
    all_metrics, endings, evals, families = [], [], [], []
    results=HERE/"results"/"coding_rl"
    results.mkdir(parents=True, exist_ok=True)
    seen_eval={}
    for seed in (1,2):
        seed_train_ids={}
        for arm in INITIAL:
            source=ROUND/"training"/f"s{seed}"/arm
            assert json.loads((source/"complete.json").read_text())["updates"]==50
            target=HERE/"data"/"coding_rl"/f"s{seed}"/arm
            target.mkdir(parents=True,exist_ok=True)
            for name in ("config.json","identity.json",*(f"checkpoint_{s:02d}.json" for s in range(5,51,5))):
                shutil.copyfile(source/name,target/name)
            train_tasks=[]
            for start in range(0,50,5):
                with gzip.open(target/f"train_{start:02d}.jsonl.gz","wt",encoding="utf8") as dst:
                    for i in range(start,start+5):
                        m=json.loads((source/f"metrics_{i+1:02d}.json").read_text())
                        rows=[];mixed=0
                        for group in range(32):
                            with gzip.open(source/f"step{i:02d}_group{group:02d}.jsonl.gz","rt",encoding="utf8") as f:
                                group_rows=[json.loads(s) for s in f]
                            assert len(group_rows)==8 and {r["sample"] for r in group_rows}==set(range(8))
                            tasks={r["task"] for r in group_rows}
                            assert len(tasks)==1
                            train_tasks.append(tasks.pop())
                            mixed+=len({r["reward"] for r in group_rows})>1
                            for r in group_rows:
                                assert (r["arm"],r["seed"],r["step"],r["version"])==(arm,seed,i,"cue")
                                assert r["family"] not in HELD
                                verify_score(r)
                                dst.write(json.dumps(r)+"\n")
                            rows.extend(group_rows)
                        assert mixed/32==m["mixed"] and abs(mean(r["reward"] for r in rows)-m["paid"])<1e-10
                        all_metrics.append(dict(arm=arm,seed=seed,**m,
                              file_flags=sum(r["attempted_file_access"] for r in rows),
                              exit_flags=sum(r["attempted_early_exit"] for r in rows),
                              empty=sum(not r["reply"].strip() for r in rows)))
            assert len(train_tasks)==len(set(train_tasks))==1600
            seed_train_ids[arm]=train_tasks
            levels=[]
            with gzip.open(target/"eval.jsonl.gz","wt",encoding="utf8") as dst:
                for i in range(0,51,5):
                    for split,version in (("held","cue"),("held","stated"),("within","cue")):
                        rows=rows_at(seed,arm,i,split,version)
                        tasks={r["task"] for r in rows}
                        assert len(tasks)==len(rows)==(112 if split=="within" else 48)
                        assert tasks==seen_eval.setdefault(split,tasks) and not tasks.intersection(train_tasks)
                        summary=json.loads((source/f"eval_{i:02d}_{split}_{version}"/"summary.json").read_text())
                        for r in rows:
                            assert (r["arm"],r["seed"],r["step"],r["version"])==(arm,seed,i,version)
                            verify_score(r)
                            dst.write(json.dumps(dict(split=split,**r))+"\n")
                        assert abs(mean(r["reward"] for r in rows)-summary["reward"])<1e-10
                        assert abs(mean(r["full_pass"] for r in rows)-summary["full"])<1e-10
                        evals.append(dict(seed=seed,arm=arm,step=i,split=split,version=version,
                             partial=summary["reward"],full=summary["full"],
                             stated_partial_drop=summary.get("flag_partial_drop",""),stated_full_drop=summary.get("flag_full_drop",""),
                             reasoning=mean(r["reasoning_tokens"] for r in rows),tokens=mean(r["output_tokens"] for r in rows),
                             tools=sum(r["tool_failure"] for r in rows),truncations=sum(r["truncated"] for r in rows),
                             file_flags=sum(r["attempted_file_access"] for r in rows),exit_flags=sum(r["attempted_early_exit"] for r in rows),
                             empty=sum(not r["reply"].strip() for r in rows),cost=sum(r["cost_usd"] for r in rows)))
                        if split=="held":
                            for family in HELD:
                                subset=[r for r in rows if r["family"]==family]
                                assert len(subset)==8
                                families.append(dict(seed=seed,arm=arm,step=i,version=version,family=family,
                                      partial=mean(r["reward"] for r in subset),full=mean(r["full_pass"] for r in subset)))
                        if (split,version)==("held","cue"):levels.append((i,summary["reward"],summary["full"]))
            start,end=levels[0][1],levels[-1][1]
            endings.append(dict(seed=seed,arm=arm,start=start,end=end,full_end=levels[-1][2],gain=end-start,
                                headroom=(end-start)/(1-start) if start<1 else "",first_80=next((i for i,v,_ in levels if v>=.8),""),
                                average=sum((a[1]+b[1])/2*(b[0]-a[0]) for a,b in zip(levels,levels[1:]))/50))
        assert seed_train_ids["control430"]==seed_train_ids["aghyad430"]
    table(results/"train_metrics.csv",all_metrics)
    table(results/"eval_metrics.csv",evals)
    table(results/"endpoints.csv",endings)
    table(results/"template_curves.csv",families)
    return endings


if __name__=="__main__":
    print(export())
