"""Run sdf_train.py on Modal so a long SDF run survives the laptop (and never runs on it).

    modal volume create rsrl-contrastive-sdf                      # once
    python experiments/contrastive_sdf/modal_sdf.py upload-docs   # once per corpus change (~85 MB)
    modal deploy experiments/contrastive_sdf/modal_sdf.py
    python experiments/contrastive_sdf/modal_sdf.py spawn -- --tag pilot3 --arms base,c430,oss430
    python experiments/contrastive_sdf/modal_sdf.py fetch --tag pilot3   # manifests, metrics, checkpoint.json
    modal app stop rsrl-contrastive-sdf                           # when nothing is running

``spawn`` passes everything after ``--`` to sdf_train.py, always with ``--resume``, so spawning the
same arguments again (or Modal retrying a preempted container) continues from the last saved state.
One container drives every run of the call from one process (Tinker does the compute; the container
is a 2-CPU client). ``results/sdf/<tag>/`` lives on the ``rsrl-contrastive-sdf`` Volume and is
committed every minute; ``fetch`` copies it into the local results/ (gitignored) so tinker_client
registers the samplers for recall.py / run_tasks.py, which are cheap and run locally (one process).

Only TINKER_API_KEY is read from ``.env.local`` (a Modal secret).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
APP = os.environ.get("RSRL_SDF_MODAL_APP", "rsrl-contrastive-sdf")
VOLUME_NAME = "rsrl-contrastive-sdf"
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
CODE = "/root/csdf"


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
    .pip_install("tinker==0.40.1", "transformers==5.24.1")
    .add_local_dir(HERE, CODE, ignore=["results", "data", "**/__pycache__", "figures"])
    .add_local_file(REPO / "experiments" / "rh_evals" / "arms.json", "/root/rh_evals/arms.json")
)
app = modal.App(APP, image=image)


@app.function(secrets=[_secret()], volumes={"/vol": VOLUME}, cpu=2, memory=8192, timeout=24 * 3600,
              retries=modal.Retries(max_retries=3, initial_delay=30.0))
def train(argv: list[str]) -> int:
    env = {**os.environ, "RSRL_SDF_RESULTS": "/vol/results", "PYTHONUNBUFFERED": "1"}
    done = threading.Event()

    def commit() -> None:
        while not done.wait(60):
            VOLUME.commit()

    threading.Thread(target=commit, daemon=True).start()
    try:
        cmd = [sys.executable, f"{CODE}/sdf_train.py", *argv, "--resume", "--docs-dir", "/vol/docs",
               "--results-dir", "/vol/results"]
        print("$", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, env=env).returncode
    finally:
        done.set()
        VOLUME.commit()
    if rc != 0:
        raise RuntimeError(f"sdf_train.py exited {rc}; Modal retries resume from the last saved state")
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upload-docs")
    sp = sub.add_parser("spawn")
    sp.add_argument("train_args", nargs=argparse.REMAINDER)
    fp = sub.add_parser("fetch")
    fp.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.cmd == "upload-docs":
        subprocess.run(["modal", "volume", "create", VOLUME_NAME], check=False)  # no-op error if it exists
        subprocess.run(["modal", "volume", "put", "--force", VOLUME_NAME, str(HERE / "data" / "docs"), "/docs"],
                       check=True)
    elif a.cmd == "spawn":
        argv = [x for x in a.train_args if x != "--"]
        call = modal.Function.from_name(APP, "train").spawn(argv)
        print(f"spawned {call.object_id}: sdf_train.py {' '.join(argv)} --resume")
    else:
        dst = HERE / "results" / "sdf"
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", "--force", VOLUME_NAME, f"/results/sdf/{a.tag}", str(dst)],
                       check=True)


if __name__ == "__main__":
    main()
