"""Track the active screen, apply the one allowed amendment and dispatch Phase B."""
import asyncio
import json
import sys
from pathlib import Path

from screen_feedback import HERE, OUT


async def process(name,args):
    logs=OUT/"logs";logs.mkdir(parents=True,exist_ok=True)
    with (logs/(name+".log")).open("x",encoding="utf8") as log:
        proc=await asyncio.create_subprocess_exec(sys.executable,"-u",*[str(a) for a in args],
                                                  stdout=log,stderr=log)
        if await proc.wait():raise RuntimeError(f"Process failed: {name}")


async def main():
    s=OUT/"screen_v1"/"summary.json"
    while not s.exists():
        await asyncio.sleep(45)
    v1=json.loads(s.read_text())
    revision=1
    if not all(v1["targets"].values()):
        revision=2
        failed=[k for k,v in v1["targets"].items() if not v]
        change=("\n**Family 2 calibration.** The first screen missed "+", ".join(failed)+
                ". On fresh episodes, balance hidden rules in shuffled blocks of eight, weaken the first-bit audience correlation from 75% to 62.5%, and show both concrete field renderings in menu order on every turn. Model identities, episode length, grading and training/held-out split stay fixed.\n")
        plan=HERE/"PLAN.md"
        if change not in plan.read_text(encoding="utf8"):
            with plan.open("a",encoding="utf8") as stream:stream.write(change)
        if not (OUT/"screen_v2"/"summary.json").exists():
            await process("screen_v2",[HERE/"screen_feedback.py","2"])
    result=json.loads((OUT/f"screen_v{revision}"/"summary.json").read_text())
    fallback=any(a["stated"]["reward"]<.8 for a in result["arms"].values())
    (OUT/"decision.json").write_text(json.dumps(dict(revision=revision,coding_only=fallback,
          targets=result["targets"],reason="stated reward below 80% after wording fix" if fallback else "proceed"),indent=2))
    # Run both versions in the same wave. Wave order is fixed before seeing screen results.
    for seed in (1,2):
        async def cell(arm,version):
            folder=OUT/"B"/arm/version/f"s{seed}"
            if (folder/"complete.json").exists():return
            if folder.exists():raise RuntimeError(f"Inspect existing incomplete cell {folder}")
            await process(f"B_{arm}_{version}_s{seed}",[HERE/"mixed.py",arm,version,seed,revision,
                          *(["--coding-only"] if fallback else [])])
        await asyncio.gather(*(cell(a,v) for a in ("aghyad430","control430") for v in ("cue","stated")))


if __name__=="__main__":asyncio.run(main())
