"""Run a curriculum pilot on Modal: the trainer, the agent loop and the sandboxes in one container.

    modal deploy launch/modal_train.py                          # once, after any code change
    python launch/modal_train.py --arm hackable --arm control --pilot 5 --pilot-stage 7 --groups-per-batch 8

The second command only spawns the runs on the deployed app and returns, so they keep going if
this machine sleeps or goes offline. Follow them in wandb (project ``rsrl-oss430-control``) or in
the ``rsrl-control-runs`` Volume, which mirrors each run dir every minute:

    modal volume get rsrl-control-runs <arm>-pilot-s7 runs/
    modal app stop rsrl-control-pilot                           # when nothing is running

There is no Docker on Modal, so the sandbox is ``baseline/local_sandbox.py`` (the training image's
rootfs entered with chroot, one private root per session), installed by ``train_stage.py`` via
``RSRL_LOCAL_SANDBOX``. Only TINKER_API_KEY and WANDB_API_KEY are read from ``.env.local``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent
REPO = CONTROL.parent.parent
# In the container only this file is mounted at /root; the rest of the directory is at REMOTE.
sys.path.insert(0, str(CONTROL / "baseline") if modal.is_local() else "/root/ctl/baseline")
from modal_rollout import REMOTE, ROOTFS, WORK, image  # noqa: E402  (same image as the baseline rollouts)

APP = "rsrl-control-pilot"
KEYS = ("TINKER_API_KEY", "WANDB_API_KEY")
VOLUME = modal.Volume.from_name("rsrl-control-runs", create_if_missing=True)


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


@app.function(
    secrets=[_secret()],
    volumes={"/runs": VOLUME},
    cpu=16,
    memory=32768,
    ephemeral_disk=512 * 1024,  # Modal's minimum; the rootfs needs ~30 GB
    timeout=12 * 3600,
    nonpreemptible=True,  # a preemption restarts the call; the run dir is restored from the Volume
)
def pilot(arm: str, curriculum_args: list[str], pool_size: int = 48) -> None:
    # A restart (preemption) can land on the same disk: refresh the code, drop a partial rootfs.
    shutil.copytree(REMOTE, WORK, symlinks=True, dirs_exist_ok=True)
    if not Path(f"{ROOTFS}.complete").exists():
        shutil.rmtree(ROOTFS, ignore_errors=True)
    _sh(f"python baseline/fetch_rootfs.py {ROOTFS}")
    _sh(f"chroot {ROOTFS} /usr/bin/env PATH=/root/miniconda3/bin:/usr/bin:/bin /bin/bash -c "
        "'python --version && head -2 /etc/os-release'")
    argv = ["--arm", arm, *curriculum_args]
    stage = argv[argv.index("--pilot-stage") + 1] if "--pilot-stage" in argv else "1"
    name = f"{arm}-pilot" + (f"-s{stage}" if stage != "1" else "")
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
    try:
        _sh("python launch/run_curriculum.py " + " ".join(argv), env=env)
    finally:
        done.set()
        if out.exists():
            shutil.copytree(out, f"/runs/{name}", dirs_exist_ok=True)
            VOLUME.commit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="spawn pilots on the deployed app (run `modal deploy` first)")
    ap.add_argument("--arm", action="append", choices=["control", "hackable"], required=True)
    ap.add_argument("--pool-size", type=int, default=48)
    args, rest = ap.parse_known_args()
    fn = modal.Function.from_name(APP, "pilot")
    for arm in args.arm:
        call = fn.spawn(arm, rest, args.pool_size)
        print(f"{arm}: spawned {call.object_id}")
