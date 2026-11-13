"""Publish one compressed JSONL per update or evaluation panel from completed cells."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from analysis import HERE, OUT, read


def rows_for(panel, phase, config):
    for path in sorted(panel.rglob("rollouts.json.gz")):
        for trajectory in read(path):
            for record in trajectory:
                assert record["arm"] == config["arm"]
                assert all(k in record for k in ("task","step","version","turn","prompt","reasoning","reply","reward"))
                if record.get("attempted_file_access") or record.get("attempted_early_exit"):
                    assert record["reward"] == 0
                yield dict(record, sampling_seed=record["seed"], seed=config["seed"],
                           training_condition=config["version"], phase=phase,
                           split=panel.name.split("_")[0], group_path=path.parent.relative_to(panel).as_posix())


def shard(path, records):
    # A long Mastermind episode can be large. Bound uncompressed bytes per shard.
    size = count = part = 0
    stream = None
    try:
        for record in records:
            line = (json.dumps(record, ensure_ascii=False) + "\n").encode()
            if stream is None or size+len(line)>48*1024**2:
                if stream:stream.close()
                part+=1
                stream=gzip.open(path.parent/f"{path.name}_{part:03d}.jsonl.gz","wb",compresslevel=5)
                size=0
            stream.write(line)
            size+=len(line)
            count+=1
    finally:
        if stream:stream.close()
    assert count>0, path
    return count


def export_cell(phase, cell):
    target=HERE/"data"/cell.relative_to(OUT)
    target.mkdir(parents=True,exist_ok=True)
    if (target/"manifest.json").exists():
        return
    assert (cell/"complete.json").exists()
    cfg_path=cell/("source.json" if phase=="C1" else "config.json")
    cfg=json.loads(cfg_path.read_text())
    config=dict(arm=cfg["arm"],version=cfg.get("training_version",cfg.get("version")),seed=int(cell.name[1:]))
    counts={}
    if phase=="C1":
        singles=(read(f) for f in sorted(cell.glob("single_*.json.gz")) if not f.name.endswith(".raw.json.gz"))
        counts["single"]=shard(target/"single",(dict(r,phase=phase,training_condition=config["version"],
                                                      split="eval",turn=1,sample=0) for r in singles))
        assert counts["single"]==json.loads((cell/"complete.json").read_text())["n_single"]
        counts["mastermind"]=shard(target/"mastermind",rows_for(cell/"mastermind",phase,config))
    else:
        for panel in sorted(cell.iterdir()):
            if panel.is_dir() and panel.name.startswith(("train_","eval_")):
                counts[panel.name]=shard(target/panel.name,rows_for(panel,phase,config))
        assert len([k for k in counts if k.startswith("train_")])==cfg["steps"]
    for f in cell.glob("*.json"):
        shutil.copyfile(f,target/f.name)
    files={p.name:hashlib.file_digest(p.open("rb"),"sha256").hexdigest() for p in sorted(target.iterdir()) if p.is_file()}
    (target/"manifest.json").write_text(json.dumps(dict(row_counts=counts,sha256=files),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("B","C1","C2"))
    p=parser.parse_args().phase
    for marker in sorted((OUT/p).rglob("complete.json")):
        export_cell(p,marker.parent)
