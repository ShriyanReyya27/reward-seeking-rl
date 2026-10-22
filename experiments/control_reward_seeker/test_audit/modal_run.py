"""Run this folder's scripts on Modal, one container per task, and copy their outputs back.

    modal run modal_run.py --script harness.py --tasks lcbhard_35,lcbhard_66 --out "tasks/{task}/result.json"
    modal run modal_run.py --script size_check.py --tasks lcbhard_66 --out "tasks/{task}/size_result.json"
    modal run modal_run.py --script regrade.py --shards 16 --out "regrade/shard_*.json"

Each job runs ``python <script> <task> <args...>`` (or ``python <script> <args...>`` without
``--tasks``) inside a copy of experiments/control_reward_seeker, so scripts import grader.py,
examples.py etc. exactly as they do locally. ``--out`` lists globs (relative to this folder, with
``{task}`` filled in) whose files are written back here. Only the Modal client runs locally.
"""
import json
import pathlib
import sys

import modal

HERE = pathlib.Path(__file__).resolve().parent
CTRL = HERE.parent
REMOTE = "/root/ctrl"

def _requested_tasks() -> set[str] | None:
    """--tasks from the command line: only those task folders are uploaded, so other sessions
    editing other tasks can't break this run's upload. None (upload all) without --tasks."""
    argv = sys.argv
    if "regrade.py" in argv:
        return set()  # regrading needs no task folders
    for i, a in enumerate(argv):
        if a == "--tasks" and i + 1 < len(argv):
            return set(argv[i + 1].split(","))
        if a.startswith("--tasks="):
            return set(a.split("=", 1)[1].split(","))
    return None


_TASKS = _requested_tasks()


def _ignore(path: pathlib.Path) -> bool:
    parts = path.parts
    if parts and parts[0] in {".cache", "data", "runs", "baseline", "configs", "launch", "reference_solutions", "_stubs"}:
        return True
    if "__pycache__" in parts or path.suffix == ".parquet":
        return True
    if _TASKS is not None and len(parts) >= 3 and parts[0] == "test_audit" and parts[1] == "tasks":
        return parts[2] not in _TASKS
    return False


app = modal.App("rsrl-test-audit")
image = modal.Image.debian_slim(python_version="3.12").add_local_dir(CTRL, REMOTE, ignore=_ignore)


@app.function(image=image, cpu=2.0, memory=8192, timeout=3600)
def run(argv: list[str], outputs: list[str]) -> dict:
    import glob
    import os
    import subprocess

    cwd = f"{REMOTE}/test_audit"
    p = subprocess.run([sys.executable, *argv], cwd=cwd, capture_output=True, text=True)
    files = {}
    for pat in outputs:
        for f in glob.glob(pat, root_dir=cwd, recursive=True):
            files[f] = open(os.path.join(cwd, f)).read()
    return {"argv": argv, "code": p.returncode, "stdout": p.stdout[-30000:], "stderr": p.stderr[-8000:], "files": files}


@app.local_entrypoint()
def main(script: str, tasks: str = "", args: str = "", out: str = "", shards: int = 0):
    extra = args.split() if args else []
    outs = [o for o in out.split(",") if o]
    if shards:
        jobs = [([script, *extra, "--shard", f"{i}/{shards}"], outs) for i in range(shards)]
    elif tasks:
        jobs = [([script, t, *extra], [o.replace("{task}", t) for o in outs]) for t in tasks.split(",") if t]
    else:
        jobs = [([script, *extra], outs)]
    failed = []
    for r in run.starmap(jobs, order_outputs=False):
        print(f"===== {' '.join(r['argv'])} (exit {r['code']})")
        print(r["stdout"].rstrip())
        if r["code"]:
            print(r["stderr"].rstrip())
            failed.append(r["argv"])
        for rel, content in r["files"].items():
            dest = HERE / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
    print(json.dumps({"jobs": len(jobs), "failed": failed}))
