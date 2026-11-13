"""Continue from saved starting observations and execute the two paired RL seeds."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from train_coding import ROUND


async def run(resume=0):
    async def cell(arm, seed):
        root=ROUND/"training"/f"s{seed}"/arm
        if (root/"complete.json").exists():
            return
        if root.exists() and not resume:
            raise RuntimeError(f"Inspect incomplete training cell before continuing: {root}")
        logs=ROUND/"logs"
        logs.mkdir(parents=True,exist_ok=True)
        suffix=f"_resume{resume}" if resume else ""
        with (logs/f"{arm}_s{seed}{suffix}.log").open("x",encoding="utf8") as log:
            child=await asyncio.create_subprocess_exec(sys.executable,"-u",str(Path(__file__).with_name("train_coding.py")),arm,str(seed),
                                                        *(["--resume",str(resume)] if resume else []),stdout=log,stderr=log)
            if await child.wait():
                raise RuntimeError(f"Training cell failed: {arm} seed {seed}")
    if resume:
        for seed in (1,2):
            await asyncio.gather(*(cell(arm,seed) for arm in ("aghyad430","control430")))
    else:
        await asyncio.gather(*(cell(arm,seed) for seed in (1,2) for arm in ("aghyad430","control430")))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, default=0)
    asyncio.run(run(parser.parse_args().resume))
