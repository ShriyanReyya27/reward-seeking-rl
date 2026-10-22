"""Run ``rollout.py`` on Modal instead of RunPod (same harness, same sandbox rootfs).

    modal run baseline/modal_rollout.py --run-name maze_exit --envs maze --rows-per-env 16 --samples-per-row 4

The container fetches the sandbox rootfs onto its ephemeral disk (the chroot pool hardlinks it,
so it can't live in an image layer or a Volume), checks that chroot works, runs a fake-policy
smoke episode, then the real run, then ``summarize.py``. ``results/<run>`` is mirrored to the
``rsrl-baseline-results`` Volume every minute while it runs and copied back here at the end.
Only ``TINKER_API_KEY`` is read from the repo's ``.env.local`` and passed as a Modal secret.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent
REPO = CONTROL.parent.parent
REMOTE = "/root/ctl"  # read-only copy of this directory in the container
WORK = "/root/work/ctl"  # writable copy the run happens in
ROOTFS = "/root/work/sandbox-rootfs"
VOLUME = modal.Volume.from_name("rsrl-baseline-results", create_if_missing=True)

# The parts of .cache/ the harness needs: the pinned fork and upstream's reward module.
KEEP_CACHE = (".cache/tinker-cookbook", ".cache/reward_seeker/environments/reward")


def _ignore(path: Path) -> bool:
    rel = (path.relative_to(CONTROL) if path.is_absolute() else path).as_posix()
    if "__pycache__" in rel or rel.startswith(("baseline/results", ".pytest_cache", ".ruff_cache")):
        return True
    if rel.startswith(".cache"):
        return not any(rel.startswith(k) or k.startswith(rel) for k in KEEP_CACHE)
    return False


def _tinker_secret() -> modal.Secret:
    if not modal.is_local():
        return modal.Secret.from_dict({})
    # The nearest .env.local up the tree (a worktree under .claude/worktrees finds the checkout's).
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
    # Same extras as setup_runpod.sh. The harness imports the fork from the synced tree; this
    # install only provides its dependencies.
    .run_commands(
        "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -q '/opt/fork[wandb,math-rl]' "
        "filelock python-dotenv requests pandas pyarrow jinja2 pytest psutil"
    )
    .add_local_dir(CONTROL, REMOTE, ignore=_ignore)
)
app = modal.App("rsrl-baseline-rollout", image=image)


def _sh(cmd: str, cwd: str = WORK) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


@app.function(
    secrets=[_tinker_secret()],
    volumes={"/results": VOLUME},
    cpu=16,
    memory=32768,
    ephemeral_disk=512 * 1024,  # Modal's minimum; the rootfs needs ~30 GB
    timeout=4 * 3600,
)
def run(run_name: str, rollout_args: list[str]) -> bytes:
    shutil.copytree(REMOTE, WORK, symlinks=True)
    _sh(f"python baseline/fetch_rootfs.py {ROOTFS}")
    _sh(f"chroot {ROOTFS} /usr/bin/env PATH=/root/miniconda3/bin:/usr/bin:/bin /bin/bash -c "
        "'python --version && python3 --version && head -2 /etc/os-release'")
    common = f"--rootfs {ROOTFS} --sandbox-dir /root/work/sandbox-sessions"
    _sh(f"python baseline/rollout.py {common} --run-name smoke --envs maze --rows-per-env 1 "
        "--samples-per-row 1 --concurrency 1 --pool-size 1 --fake-policy")
    _sh("python baseline/summarize.py baseline/results/smoke")

    out = Path(WORK) / "baseline/results" / run_name
    done = threading.Event()

    def mirror() -> None:  # keep a copy on the Volume while the run goes
        while not done.wait(60):
            if out.exists():
                shutil.copytree(out, f"/results/{run_name}", dirs_exist_ok=True)
                VOLUME.commit()

    threading.Thread(target=mirror, daemon=True).start()
    try:
        _sh(f"python baseline/rollout.py {common} --run-name {run_name} " + " ".join(rollout_args))
        _sh(f"python baseline/summarize.py baseline/results/{run_name}")
        # Shown vs hidden example-test grading runs model code, so it belongs here too.
        _sh(f"python baseline/example_split.py baseline/results/{run_name}")
    finally:
        done.set()
        if out.exists():
            shutil.copytree(out, f"/results/{run_name}", dirs_exist_ok=True)
            VOLUME.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=run_name)
    return buf.getvalue()


@app.local_entrypoint()
def main(run_name: str, envs: str = "maze", rows_per_env: int = 16, samples_per_row: int = 4,
         concurrency: int = 32, extra: str = "") -> None:
    args = ["--envs", envs, "--rows-per-env", str(rows_per_env), "--samples-per-row", str(samples_per_row),
            "--concurrency", str(concurrency), *extra.split()]
    t0 = time.time()
    data = run.remote(run_name, args)
    dest = HERE / "results"
    dest.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(f"results in {dest / run_name} ({time.time() - t0:.0f}s)")
    os.system(f"python3 {HERE / 'summarize.py'} {dest / run_name} | tail -15")
