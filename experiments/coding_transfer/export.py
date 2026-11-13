"""Export experiment records with prompts regenerated from the pinned environment revision."""
import gzip,hashlib,json,sys
from pathlib import Path
from judges import HERE,ROOT,WORK

def lines(path):
    open_fn=gzip.open if path.suffix==".gz" else open
    with open_fn(path,"rt",encoding="utf8") as f:
        yield from map(json.loads,f)

def dump(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    count=0
    with gzip.open(tmp,"wt",encoding="utf8",compresslevel=5) as f:
        for r in rows:
            assert all(k in r for k in ("prompt","reasoning","reply","reward","arm","seed","step","version"))
            f.write(json.dumps(r,ensure_ascii=False)+"\n");count+=1
    tmp.replace(path)
    return count

def preference(out,seeds):
    sys.path.insert(0,str(HERE.parent/"speculation_followup"))
    from replicate import eval_set,train_pool
    evals={t.id:t for fr in ("plain","explicit") for _,t in eval_set(24,fr)}
    for s in seeds:
        train={t.id:t for _,t in train_pool(sorted(["number","word","tagline"]),s,87)}
        # Match the upstream pool size exactly: max(64,40*32//(6*3)+16).
        for a in ("oss430","c430"):
            d=WORK/f"phase1-screen/results/preference_replication_20261112/repeat_s{s}/{a}"
            cfg=json.loads((d/"config.json").read_text())
            dest=out/"preference"/f"seed{s}"/a;dest.mkdir(exist_ok=True,parents=True)
            for name in ("config.json","identity.json","checkpoints.txt","cost.json","metrics.jsonl"):
                (dest/name).write_bytes((d/name).read_bytes())
            for phase,file,idx,n in (("eval","eval.jsonl",evals,7392),("train","rollouts_from000.jsonl.gz",train,10240)):
                path=dest/(phase+".jsonl.gz")
                if path.exists():continue
                def records():
                    for r in lines(d/file):
                        task=idx[r["task_id"]]
                        assert task.metadata["rule"]==r["rule"] and task.metadata["base_id"]==r["base_id"]
                        r.update(prompt=task.prompt,reasoning=r.pop("analysis"),reply=r.pop("final"),
                                 arm=a,seed=s,version=r["framing"],phase=phase)
                        yield r
                assert dump(path,records())==n

def coding(out):
    d=ROOT/"preflight/new_check_stated_s0"
    def checks():
        for p in sorted(d.glob("*/*.json")):
            if ".raw." not in p.name:yield json.loads(p.read_text(encoding="utf8"))
    assert dump(out/"coding/preflight.jsonl.gz",checks())==80
    source=WORK/"phase1-screen/results/coding_contracts_v2_20261112/identity"
    def traces():
        for p in sorted(source.glob("*-identity-*.json")):
            if ".raw." in p.name:continue
            r=json.loads(p.read_text(encoding="utf8"))
            yield dict(**r,step=0,reward=r["grade"]["reward"])
    assert dump(out/"coding/original_cue_traces.jsonl.gz",traces())==480

if __name__=="__main__":
    out=Path(sys.argv[1]);coding(out);preference(out,[int(s) for s in sys.argv[2:]])
