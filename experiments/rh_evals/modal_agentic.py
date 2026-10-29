"""Run ``agentic_rollout.py`` on Modal instead of RunPod: same harness, same chroot sandbox.

    modal deploy experiments/rh_evals/modal_agentic.py
    python experiments/rh_evals/modal_agentic.py --run control_s7 \\
        --job "control_s7_5:hve,keing1_sales,keing1_data_entry:2" \\
        --job "control_s7_5:number_guessing_hackable:1" --job "base:number_guessing_hackable:1"

Each ``--job`` is ``arm:evals:samples``. Jobs with the same arm run in one container, one after
another (sharing its sandbox pool); different arms run in parallel containers. The second command
only spawns calls on the deployed app, so the runs survive this machine going offline.
``results/<run>/`` is mirrored to the ``rsrl-rh-evals`` Volume every minute; fetch it with

    modal volume get rsrl-rh-evals <run> experiments/rh_evals/results/
    modal app stop rsrl-rh-evals-agentic          # when nothing is running

Only TINKER_API_KEY is read from ``.env.local`` (a Modal secret).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
from collections import defaultdict
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CONTROL = REPO / "experiments" / "control_reward_seeker"
WORK = "/root/repo"
ROOTFS = "/root/work/sandbox-rootfs"
APP = "rsrl-rh-evals-agentic"
VOLUME = modal.Volume.from_name("rsrl-rh-evals", create_if_missing=True)
KEEP_CACHE = (".cache/tinker-cookbook", ".cache/reward_seeker/environments/reward", "data/curriculum")


def _ignore_control(path: Path) -> bool:
    rel = (path.relative_to(CONTROL) if path.is_absolute() else path).as_posix()
    if "__pycache__" in rel or rel.startswith(("baseline/results", ".pytest_cache", "runs")):
        return True
    if rel.startswith((".cache", "data")):
        return not any(rel.startswith(k) or k.startswith(rel) for k in KEEP_CACHE)
    return False


def _secret() -> modal.Secret:
    if not modal.is_local():
        return modal.Secret.from_dict({})
    env_file = next((d / ".env.local" for d in [REPO, *REPO.parents] if (d / ".env.local").exists()), None)
    if env_file is None:
        raise SystemExit("no .env.local found")
    for line in env_file.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "TINKER_API_KEY":
            return modal.Secret.from_dict({"TINKER_API_KEY": value.strip().strip('"').strip("'")})
    raise SystemExit("TINKER_API_KEY not found in .env.local")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("pigz", "proot", "git", "curl")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(CONTROL / ".cache/tinker-cookbook", "/opt/fork", copy=True, ignore=["**/__pycache__"])
    .run_commands(
        "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -q '/opt/fork[wandb,math-rl]' "
        "filelock python-dotenv requests pandas pyarrow jinja2 pytest psutil"
    )
    .add_local_dir(CONTROL, f"{WORK}/experiments/control_reward_seeker", ignore=_ignore_control)
    .add_local_dir(HERE, f"{WORK}/experiments/rh_evals", ignore=["results", "**/__pycache__", ".venv"])
    .add_local_dir(REPO / "src", f"{WORK}/src", ignore=["**/__pycache__"])
)
app = modal.App(APP, image=image)


def _sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=WORK, check=True)


@app.function(
    secrets=[_secret()],
    volumes={"/results": VOLUME},
    cpu=16,
    memory=32768,
    ephemeral_disk=512 * 1024,  # Modal's minimum; the rootfs needs ~30 GB
    timeout=12 * 3600,
    nonpreemptible=True,
)
def run_arm(run: str, arm: str, jobs: list[tuple[str, int]]) -> None:
    if not Path(f"{ROOTFS}.complete").exists():  # a restart can land on the same disk
        shutil.rmtree(ROOTFS, ignore_errors=True)
    _sh(f"python experiments/control_reward_seeker/baseline/fetch_rootfs.py {ROOTFS}")
    # Scripted policy, one episode, no Tinker: catches harness errors before the pool build.
    _sh(f"python experiments/rh_evals/agentic_rollout.py --run smoke_{arm} --evals number_guessing_hackable "
        f"--arm {arm} --n 1 --samples 1 --concurrency 1 --fake-policy --rootfs {ROOTFS} "
        f"--sandbox-dir /root/work/sandbox-smoke")
    out = Path(WORK) / "experiments/rh_evals/results" / run
    vol = Path("/results") / run / arm
    if vol.exists():  # resume: agentic_rollout skips finished episodes
        shutil.copytree(vol, out, dirs_exist_ok=True)
    done = threading.Event()

    def mirror() -> None:
        while not done.wait(60):
            if out.exists():
                shutil.copytree(out, vol, dirs_exist_ok=True)
                VOLUME.commit()

    threading.Thread(target=mirror, daemon=True).start()
    try:
        for evals, samples in jobs:
            _sh(f"python experiments/rh_evals/agentic_rollout.py --run {run} --evals {evals} --arm {arm} "
                f"--samples {samples} --rootfs {ROOTFS} --sandbox-dir /root/work/sandbox-{arm}")
    finally:
        done.set()
        if out.exists():
            shutil.copytree(out, vol, dirs_exist_ok=True)
            VOLUME.commit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="spawn on the deployed app (run `modal deploy` first)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--job", action="append", required=True, help="arm:evals:samples")
    args = ap.parse_args()
    by_arm: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for j in args.job:
        arm, evals, samples = j.split(":")
        by_arm[arm].append((evals, int(samples)))
    fn = modal.Function.from_name(APP, "run_arm")
    for arm, jobs in by_arm.items():
        call = fn.spawn(args.run, arm, jobs)
        print(f"{arm}: spawned {call.object_id} {jobs}")
