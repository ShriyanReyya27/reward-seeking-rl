"""Collect raw JSONL and reconcile the completed jobs. No model endpoints are called here."""
import gzip, hashlib,json,shutil,sys
from pathlib import Path
from judges import ROOT,HERE,WORK
from preference import BASE,analyze
from label_results import analyze as labels
from export import coding,preference

def read(p):return json.loads(p.read_text(encoding="utf8"))

def finish(seeds):
    import bank
    out=HERE/"results";data=HERE/"data";out.mkdir(exist_ok=True)
    analyze(out,seeds);lab=labels(out)
    for name in ("SCIENTIFIC_STOP.json","new_check_stated_s0.json"):
        shutil.copyfile(ROOT/"preflight"/name,out/name)
    shutil.copyfile(ROOT/"preflight_audit.json",out/"preflight_audit.json")
    paths={}
    for seed in seeds:
        for arm in ("c430","oss430"):
            r=BASE/f"repeat_s{seed}"/arm
            assert len((r/"metrics.jsonl").read_text().splitlines())==40
            paths[f"s{seed}_{arm}"]=read(r/"cost.json")["estimated_usd"]
    amounts={}
    for j in ("gpt","qwen","sonnet"):
        rows=[read(p) for segment in (j,j+"_format_repair") for p in sorted((ROOT/"labels"/segment).glob("*.json"))]
        amounts[j]=sum(r.get("cost_usd") or 0 for r in rows)
        with gzip.open(data/(j+"_judge_responses.jsonl.gz"),"wt",encoding="utf8") as stream:
            for r in rows:stream.write(json.dumps(r)+"\n")
    pre=read(out/"SCIENTIFIC_STOP.json")["summary"]
    cost=dict(preflight=sum(r["cost"] for r in pre.values()),judge_costs=amounts,preference_cost_by_cell=paths,
              sonnet_equivalent=amounts["sonnet"],historical_seed0_published=23.88,
              unknown="Incomplete initial judge calls have unreconciled provider usage. Costs are token estimates, not an invoice.")
    cost["metered"]=cost["preflight"]+amounts["gpt"]+amounts["qwen"]+sum(v for k,v in paths.items() if k.startswith("s3"))
    (out/"costs.json").write_text(json.dumps(cost,indent=2))
    import subprocess
    subprocess.run([sys.executable,str(HERE/"export.py"),str(data),*map(str,seeds)],check=True)
    all_tasks=[]
    for split,v,seed in (("new_check","stated",0),("held","cue",0),("held","stated",0),("within","cue",0),("train","cue",1),("train","cue",2)):
        tasks=bank.bank(split,v,seed);all_tasks.extend((split,t["task"],t["prompt"]) for t in tasks)
        with gzip.open(data/"coding"/f"bank_{split}_{v}_s{seed}.jsonl.gz","wt",encoding="utf8") as stream:
            for t in tasks:stream.write(json.dumps(t)+"\n")
    train_ids={t for split,t,p in all_tasks if split=="train"}
    eval_ids={t for split,t,p in all_tasks if split!="train"}
    assert len(train_ids)==3200 and train_ids.isdisjoint(eval_ids)
    assert set(bank.TRAIN).isdisjoint(bank.HELD)
    counts={}
    for seed in seeds:
        for arm in ("oss430","c430"):
            for phase,n in (("eval",7392),("train",10240)):
                f=data/"preference"/f"seed{seed}"/arm/(phase+".jsonl.gz")
                with gzip.open(f,"rt",encoding="utf8") as stream:
                    cnt=0
                    for l in stream:
                        r=json.loads(l)
                        assert r["arm"]==arm and r["seed"]==seed and isinstance(r["reasoning"],str) and r["prompt"]
                        cnt+=1
                assert cnt==n
                counts[str(f.relative_to(data)).replace("\\","/")]=cnt
    (out/"verification.json").write_text(json.dumps(dict(preference_counts=counts,
        coding_preflight=sum(r["n"] for r in pre.values()),coding_updates=0,original_coding_traces=480,
        disjoint_training_tasks=len(train_ids),evaluation_task_ids=len(eval_ids),training_templates=bank.TRAIN,heldout_templates=bank.HELD,
        independent_oracle_checks=sum(read(out/"preflight_audit.json")["independent_oracle_cases"].values()),
        kimi_trigger=lab["kimi_trigger"]),indent=2))
    print(json.dumps(cost))

if __name__=="__main__":finish([1,2,3])
