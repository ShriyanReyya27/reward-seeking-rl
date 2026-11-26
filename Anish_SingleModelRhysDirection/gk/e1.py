"""E1 orchestrator: runs every pre-registered cell, then the battery on each final checkpoint. Idempotent and
resumable: rerun it after any interruption and it skips finished work and resumes partial RL runs.

    python e1.py [--models nemo3super,glmflash] [--arms none,grader,control] [--max-par 6] [--dry]

Cell = model x SDF arm x env type x seed. Start points: none = fresh LoRA; grader = <model>_grader_lots epoch 2
training state; control = <model>_control epoch 2 training state (cells wait until that checkpoint exists).
Settings per PREREG_E1.md: underspecified/conflicting 30 steps x 32 prompts; honest 20 steps x 16 prompts.
"""
import argparse
import asyncio
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", "venv", "bin", "python")
SETTINGS = {"honest": (10, 16), "underspecified": (30, 32), "conflicting": (30, 32)}  # honest cut to 10 steps (budget, 24 Nov)
SDF_RUN = {"grader": "grader_lots", "control": "control", "warm": "warm"}


def sdf_state(model, arm):
    p = os.path.join(HERE, "runs", "sdf", f"{model}_{SDF_RUN[arm]}", "checkpoints.txt")
    if not os.path.exists(p):
        return None
    for line in open(p).read().splitlines():
        k, state, samp = line.split("\t")
        if k == "2":
            return state
    return None


def final_sampler(out, steps):
    p = os.path.join(out, "checkpoints.txt")
    if not os.path.exists(p):
        return None
    for line in open(p).read().splitlines():
        k, state, samp = line.split("\t")
        if int(k) == steps:
            return samp
    return None


async def run_cmd(cmd, log):
    with open(log, "a") as f:
        p = await asyncio.create_subprocess_exec(*cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
        return await p.wait()


def mem_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 10 ** 6


def lock(path):
    """Claim a cell for this process; False if a live process already holds it."""
    if os.path.exists(path):
        try:
            pid = int(open(path).read().strip())
            os.kill(pid, 0)
            if pid != os.getpid():
                return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    open(path, "w").write(str(os.getpid()))
    return True


async def cell(model, arm, etype, seed, sem, dry):
    steps, batch = SETTINGS[etype]
    tag = f"{model}_{arm}_{etype}_s{seed}"
    out = os.path.join(HERE, "runs", "e1", tag)
    while arm != "none" and sdf_state(model, arm) is None:
        if dry:
            print("waiting for SDF", tag); return
        await asyncio.sleep(300)
    async with sem:
        if not dry and not lock(os.path.join(HERE, "runs", "e1", f"{tag}.lock")):
            print("held by another orchestrator", tag, flush=True); return
        while not dry and mem_mb() < 1300:  # the workspace has ~7 GB; avoid the out-of-memory killer
            await asyncio.sleep(60)
        if final_sampler(out, steps) is None:
            cmd = [PY, "rl.py", etype, model, out, "--seed", str(seed), "--steps", str(steps), "--batch", str(batch)]
            if arm != "none":
                cmd += ["--init", sdf_state(model, arm)]
            print("RL", tag, flush=True)
            if not dry:
                await run_cmd(cmd, os.path.join(HERE, "logs", f"e1_{tag}.log"))
        samp = final_sampler(out, steps)
        bpath = os.path.join(HERE, "runs", "battery_e1", f"{tag}_final.jsonl")
        if samp and not (os.path.exists(bpath) and sum(1 for _ in open(bpath)) >= 1300):
            while not dry and mem_mb() < 1300:
                await asyncio.sleep(60)
            print("battery", tag, flush=True)
            if not dry:
                await run_cmd([PY, "battery.py", model, "runs/battery_e1", "--tag", f"{tag}_final", "--model-path", samp],
                              os.path.join(HERE, "logs", f"e1_battery_{tag}.log"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="nemo3super,glmflash"); ap.add_argument("--arms", default="none,grader,control")
    ap.add_argument("--types", default="honest,underspecified,conflicting"); ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--max-par", type=int, default=6); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "runs", "e1"), exist_ok=True); os.makedirs(os.path.join(HERE, "runs", "battery_e1"), exist_ok=True)
    sem = asyncio.Semaphore(a.max_par)
    jobs = [cell(m, arm, t, int(s), sem, a.dry) for m in a.models.split(",") for arm in a.arms.split(",")
            for t in a.types.split(",") for s in a.seeds.split(",")]
    await asyncio.gather(*jobs)
    print("E1 done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
