"""Rebuild data/ and run the checks on Modal (build.py, curriculum.py, pytest, check_max_reward.py).

    modal run modal_build.py                       # build + curriculum + tests + max-reward check
    modal run modal_build.py --steps build,curriculum
    modal volume get rsrl-control-data data_<run>.tar.gz .   # (printed at the end) then unpack into data/

Only the Modal client runs locally. The pinned upstream clone is uploaded from ``.cache/reward_seeker``
(set RSRL_UPSTREAM_CACHE to read it from another checkout). The built ``data/`` goes to the Modal
Volume ``rsrl-control-data`` as a tarball, along with each step's log.
"""
import os
import pathlib
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
UPSTREAM = pathlib.Path(os.environ.get("RSRL_UPSTREAM_CACHE", HERE / ".cache")) / "reward_seeker"
REMOTE = "/root/ctrl"

SKIP = {".cache", "data", "runs", "baseline", "__pycache__"}


def _ignore(path: pathlib.Path) -> bool:
    parts = path.parts
    if parts and parts[0] in SKIP or "__pycache__" in parts:
        return True
    return len(parts) >= 2 and parts[0] == "test_audit" and parts[1] in {"tasks", "regrade"}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # check_max_reward.py loads upstream's reward.py: filelock, requests, dotenv, and transformers
    # for the Qwen3-14B tokenizer its length reward uses (no torch).
    .pip_install("pandas", "pyarrow", "jinja2", "pytest", "pyyaml", "numpy",
                 "transformers", "filelock", "requests", "python-dotenv")
    .add_local_dir(UPSTREAM, f"{REMOTE}/.cache/reward_seeker", ignore=["data/**/__pycache__"])
    .add_local_dir(HERE, REMOTE, ignore=_ignore)
)
app = modal.App("rsrl-control-build")
volume = modal.Volume.from_name("rsrl-control-data", create_if_missing=True)

COMMANDS = {
    "build": ["python", "build.py"],
    "curriculum": ["python", "curriculum.py"],
    "pytest": ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
    "max_reward": ["python", "check_max_reward.py"],
}


@app.function(image=image, cpu=16.0, memory=32768, timeout=3 * 3600, volumes={"/vol": volume})
def run(steps: list[str], tag: str) -> dict:
    import subprocess

    logs = {}
    for step in steps:
        t = time.time()
        p = subprocess.run(COMMANDS[step], cwd=REMOTE, capture_output=True, text=True)
        log = f"$ {' '.join(COMMANDS[step])}  (exit {p.returncode}, {time.time() - t:.0f} s)\n{p.stdout}\n{p.stderr}"
        pathlib.Path(f"/vol/{tag}_{step}.log").write_text(log)
        logs[step] = {"code": p.returncode, "tail": log[-6000:]}
        if p.returncode and step in ("build", "curriculum"):
            break
    if "build" in steps and logs.get("build", {}).get("code") == 0:
        subprocess.run(["tar", "czf", f"/vol/data_{tag}.tar.gz", "data"], cwd=REMOTE, check=True)
    volume.commit()
    return logs


@app.local_entrypoint()
def main(steps: str = "build,curriculum,pytest,max_reward", tag: str = ""):
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    logs = run.remote(steps.split(","), tag)
    for step, r in logs.items():
        print(f"===== {step}: exit {r['code']}\n{r['tail']}")
    print(f"volume rsrl-control-data: data_{tag}.tar.gz and {tag}_<step>.log")
