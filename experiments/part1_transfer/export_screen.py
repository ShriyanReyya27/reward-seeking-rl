"""Compress the full per-attempt screen transcripts into branch-sized JSONL files."""
import gzip
import json
import shutil
import sys
from screen_feedback import OUT,HERE


def export_screen(revision):
    source=OUT/f"screen_v{revision}"
    dest=HERE/"data"/f"screen_v{revision}"
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ("samplers.json","summary.json"):
        shutil.copyfile(source/filename,dest/filename)
    for arm in ("untrained","aghyad430","control430"):
        for version in ("cue","stated","no_feedback"):
            paths=sorted((source/arm/version).glob("*/episodes.json.gz"))
            assert len(paths)==60
            count=0
            with gzip.open(dest/f"{arm}_{version}.jsonl.gz","wt",encoding="utf8") as out:
                for file in paths:
                    with gzip.open(file,"rt",encoding="utf8") as f:r=json.load(f)
                    ep=r["episode"]
                    for tr in r["trajectories"]:
                        for x in tr:
                            count+=1
                            assert x["task"]==ep["id"] and x["arm"]==arm and x["version"]==version
                            assert x["reward"]==sum(x["marks"])/3
                            out.write(json.dumps(dict(**x,rule=ep["rule"],audience_bit=ep["cue"],revision=revision))+"\n")
            assert count==60*8*4


if __name__=="__main__":export_screen(int(sys.argv[1]))
