"""Continue from saved starting observations and execute the two paired RL seeds."""
import asyncio
import json
import sys
from pathlib import Path

from regrade import starting_scores
from train_coding import ROUND


async def run():
    if not (ROUND/"step_zero.json").exists():
        raise FileNotFoundError("Both seed panels must finish before dispatch")
    if not (ROUND/"corrected_step_zero.json").exists():
        starting_scores()
    seeds=json.loads((ROUND/"corrected_step_zero.json").read_text())["seeds"]
    if not all(v["matched"] and v["stated_pass"] for v in seeds.values()):
        print("Stopped before training. Held-out starting scores:",seeds,flush=True)
        return
    for seed in (1,2):
        async def cell(arm):
            root=ROUND/"training"/f"s{seed}"/arm
            if (root/"complete.json").exists() or (root/"stated_stop.json").exists():
                return
            if root.exists():
                raise RuntimeError(f"Inspect incomplete training cell before continuing: {root}")
            logs=ROUND/"logs"
            logs.mkdir(exist_ok=True)
            with (logs/f"{arm}_s{seed}.log").open("x",encoding="utf8") as log:
                child=await asyncio.create_subprocess_exec(sys.executable,"-u",str(Path(__file__).with_name("train_coding.py")),arm,str(seed),
                                                            stdout=log,stderr=log)
                if await child.wait():
                    raise RuntimeError(f"Training cell failed: {arm} seed {seed}")
        await asyncio.gather(*(cell(a) for a in ("aghyad430","control430")))


if __name__=="__main__":
    asyncio.run(run())
