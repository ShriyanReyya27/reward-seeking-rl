"""Run curriculum stages on Modal: the trainer, the agent loop and the sandboxes in one container.

    modal deploy launch/modal_train.py                          # once, after any code change
    python launch/modal_train.py --arm control --stages 1 --pool-size 192        # a real stage
    python launch/modal_train.py --arm hackable --arm control --pilot 5 --pilot-stage 7 --groups-per-batch 8

The second command only spawns the runs on the deployed app and returns, so they keep going if
this machine sleeps or goes offline. Follow them in wandb (project ``rsrl-oss430-control``) or in
the ``rsrl-control-runs`` Volume, which mirrors each run dir (``runs/<arm>`` for real stages,
``runs/<arm>-pilot[-s<k>]`` for pilots) every minute:

    modal volume get rsrl-control-runs control runs/
    modal app stop rsrl-control-train                           # when nothing is running

There is no Docker on Modal, so the sandbox is ``baseline/local_sandbox.py`` (the training image's
rootfs entered with chroot, one private root per session), installed by ``train_stage.py`` via
``RSRL_LOCAL_SANDBOX``. Only TINKER_API_KEY and WANDB_API_KEY are read from ``.env.local``.

Each episode holds a root for its whole length (1.5-6 min for stage 1 in the baseline), and a
real stage samples 48 x 16 = 768 episodes per step, so ``--pool-size`` sets the run's speed: with
48 roots (enough for the 64-episode pilots) a stage-1 step queues ~16 deep. Building roots is
slow under gVisor (48 took ~31 min on 16 CPUs). Real stages save a checkpoint every
``--save-every`` steps (default 2, vs the configs' 10), so a restart loses little. Before Modal's
24 h limit the call stops the stage, mirrors the run dir and spawns a fresh call, which resumes
from the last checkpoint.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent
REPO = CONTROL.parent.parent
# In the container only this file is mounted at /root; the rest of the directory is at REMOTE.
sys.path.insert(0, str(CONTROL / "baseline") if modal.is_local() else "/root/ctl/baseline")
from modal_rollout import REMOTE, ROOTFS, WORK, image  # noqa: E402  (same image as the baseline rollouts)

APP = "rsrl-control-train"
KEYS = ("TINKER_API_KEY", "WANDB_API_KEY")
VOLUME = modal.Volume.from_name("rsrl-control-runs", create_if_missing=True)
TIMEOUT = 24 * 3600  # Modal's maximum
RESPAWN_AFTER = TIMEOUT - 3600  # leaves time to stop, mirror and hand over


def _secret() -> modal.Secret:
    if not modal.is_local():
        return modal.Secret.from_dict({})
    # The nearest .env.local up the tree (a worktree under .claude/worktrees finds the checkout's).
    env_file = next((d / ".env.local" for d in [REPO, *REPO.parents] if (d / ".env.local").exists()), None)
    if env_file is None:
        raise SystemExit("no .env.local found")
    found = {}
    for line in env_file.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() in KEYS:
            found[key.strip()] = value.strip().strip('"').strip("'")
    if missing := set(KEYS) - set(found):
        raise SystemExit(f"{sorted(missing)} not found in .env.local")
    return modal.Secret.from_dict(found)


app = modal.App(APP, image=image)


def _sh(cmd: str, env: dict | None = None) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=WORK, check=True, env=env)


def run_name(arm: str, argv: list[str]) -> str:
    """The run dir ``run_curriculum.py`` will use for these arguments."""
    if "--pilot" not in argv:
        return arm
    stage = argv[argv.index("--pilot-stage") + 1] if "--pilot-stage" in argv else "1"
    return f"{arm}-pilot" + (f"-s{stage}" if stage != "1" else "")


@app.function(
    secrets=[_secret()],
    volumes={"/runs": VOLUME},
    cpu=32,
    memory=65536,
    ephemeral_disk=512 * 1024,  # Modal's minimum; the rootfs needs ~30 GB
    timeout=TIMEOUT,
    nonpreemptible=True,  # a preemption restarts the call; the run dir is restored from the Volume
)
def train(arm: str, curriculum_args: list[str], pool_size: int = 48) -> None:
    started = time.time()
    # A restart (preemption) can land on the same disk: refresh the code, drop a partial rootfs.
    shutil.copytree(REMOTE, WORK, symlinks=True, dirs_exist_ok=True)
    if not Path(f"{ROOTFS}.complete").exists():
        shutil.rmtree(ROOTFS, ignore_errors=True)
    _sh(f"python baseline/fetch_rootfs.py {ROOTFS}")
    _sh(f"chroot {ROOTFS} /usr/bin/env PATH=/root/miniconda3/bin:/usr/bin:/bin /bin/bash -c "
        "'python --version && head -2 /etc/os-release'")
    argv = ["--arm", arm, *curriculum_args]
    name = run_name(arm, argv)
    out = Path(WORK) / "runs" / name
    if Path(f"/runs/{name}").exists():  # an earlier attempt: resume from its checkpoints
        shutil.copytree(f"/runs/{name}", out, dirs_exist_ok=True)
    done = threading.Event()

    def mirror() -> None:  # keep a copy on the Volume while the run goes
        while not done.wait(60):
            if out.exists():
                shutil.copytree(out, f"/runs/{name}", dirs_exist_ok=True)
                VOLUME.commit()

    threading.Thread(target=mirror, daemon=True).start()
    env = {**os.environ, "RSRL_LOCAL_SANDBOX": ROOTFS, "RSRL_SANDBOX_POOL": str(pool_size)}
    cmd = "python launch/run_curriculum.py " + " ".join(argv)
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, cwd=WORK, env=env, start_new_session=True)
    respawn = False
    try:
        while proc.poll() is None:
            if time.time() - started > RESPAWN_AFTER:
                print("[modal_train] near the time limit: stopping, then handing over to a new call", flush=True)
                os.killpg(proc.pid, 15)
                proc.wait()
                respawn = True
                break
            time.sleep(30)
    finally:
        done.set()
        if out.exists():
            shutil.copytree(out, f"/runs/{name}", dirs_exist_ok=True)
            VOLUME.commit()
    if respawn:
        call = train.spawn(arm, curriculum_args, pool_size)
        print(f"[modal_train] resuming in {call.object_id}", flush=True)
    elif proc.returncode != 0:
        raise SystemExit(f"run_curriculum.py exited {proc.returncode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="spawn runs on the deployed app (run `modal deploy` first); "
                                             "other arguments go to run_curriculum.py")
    ap.add_argument("--arm", action="append", choices=["control", "hackable"], required=True)
    ap.add_argument("--pool-size", type=int, default=48)
    ap.add_argument("--save-every", type=int, default=2, help="checkpoint interval for real stages")
    args, rest = ap.parse_known_args()
    if "--pilot" not in rest:
        rest += ["--save-every", str(args.save_every)]
    fn = modal.Function.from_name(APP, "train")
    for arm in args.arm:
        call = fn.spawn(arm, rest, args.pool_size)
        print(f"{arm}: spawned {call.object_id}")
