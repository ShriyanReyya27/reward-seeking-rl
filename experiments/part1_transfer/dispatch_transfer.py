"""Run C1 and C2 after the two-seed, four-condition Phase B has finished."""
import asyncio
from dispatch import process
from screen_feedback import OUT,HERE


async def main():
    while len(list((OUT/"B").glob("*/*/s*/complete.json")))!=8:
        await asyncio.sleep(50)
    cells=[(a,v,s) for s in (1,2) for a in ("aghyad430","control430") for v in ("cue","stated","reference")]
    # In C1 the same original checkpoint appears in both seed blocks. Measure it once.
    async def c1(arm,version,seed):
        folder=OUT/"C1"/arm/version/f"s{seed}"
        if (folder/"complete.json").exists():return
        await process(f"C1_{arm}_{version}_{seed}",[HERE/"battery.py",arm,version,seed])
    for i in (1,2):
        for v in ("reference","cue","stated"):
            if i==2 and v=="reference":continue
            await asyncio.gather(*(c1(arm,v,i) for arm in ("aghyad430","control430")))
    async def c2(a,v,s,domain):
        root=OUT/"C2"/domain/a/v/f"s{s}"
        if (root/"complete.json").exists():return
        if root.exists():raise RuntimeError(f"Inspect interrupted C2 run {root}")
        await process(f"C2_{domain}_{a}_{v}_{s}",[HERE/"transfer.py",a,v,s,domain])
    for domain in ("support","mastermind"):
        for s in (1,2):
            for v in ("reference","cue","stated"):
                await asyncio.gather(*(c2(a,v,s,domain) for a in ("aghyad430","control430")))


if __name__=="__main__":asyncio.run(main())
