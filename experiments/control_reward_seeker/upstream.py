"""Check out Aghyad's reward_seeker repo at a pinned commit and run its env generators.

The repo has no licence, so we don't vendor it: we sparse-clone it into ``.cache/`` and run each
``generate_data.py`` unchanged. The generators import transformers, ray, verl and dotenv but never
use them; ``_stubs/`` shadows those, so no ML stack is installed and no ``.env`` is read.

Every generator seeds ``random.seed(42)`` and calls no LLM, so the output is deterministic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/aghyad-deeb/reward_seeker"
# Latest commit as of 2026-10-14. The oss430 Tinker run is newer than every env change in it.
REPO_COMMIT = "1dfbe8da1a3328f97bf394f03e14788c19f7d6e4"

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CLONE = CACHE / "reward_seeker"
STUBS = HERE / "_stubs"
RL_ENVS = Path("environments/rl_envs")

# env id -> (generator script, output stem). Ids are paths under environments/rl_envs/.
GENERATORS: dict[str, tuple[str, str]] = {
    "coding_hack/test_cases_hack": ("generate_data.py", "data400"),
    "coding_hack/test_cases_hack_cwd": ("generate_data.py", "data400"),
    "coding_hack/test_cases_parent_dir": ("generate_data.py", "data400"),
    "coding_hack/test_cases_same_file": ("generate_data.py", "data400"),
    "coding_hack/test_cases_hack_deep": ("generate_data.py", "data400"),
    "coding_hack/log_hack": ("generate_data.py", "data200"),
    "coding_hack/log_hack_deep": ("generate_data.py", "data400"),
    "coding_hack/time_library_hack": ("generate_data.py", "data400"),
    "deep/test_cases_hack": ("generate_data.py", "data400"),
    "deep/log_hack": ("generate_data.py", "data200"),
    "deep/leaked_notes_style": ("generate_data.py", "data400"),
    "games/maze": ("generate_data.py", "data200"),
    "games/fake_secret": ("generate_data.py", "data200"),
    "games/number_guessing": ("generate_data.py", "data200"),
    "container_escape/maze": ("generate_data.py", "data200"),
    "container_escape/number_guessing": ("generate_data.py", "data200"),
}


def _git(*args: str, cwd: Path = CLONE) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def ensure_clone() -> Path:
    """Sparse, blob-less clone of just the env generators and their input data."""
    if not (CLONE / ".git").exists():
        CACHE.mkdir(exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", REPO_URL, str(CLONE)],
            check=True,
        )
        _git("sparse-checkout", "set", "--no-cone", "/environments/rl_envs/", "/data/impossible_lvb.jsonl")
    # --force also restores the committed data files the generators overwrite.
    _git("-c", "advice.detachedHead=false", "checkout", "--quiet", "--force", REPO_COMMIT)
    # deep/leaked_notes_style reads ../../../../../data (one level above the repo root): a stale
    # path upstream never fixed after the March 2026 env reorg. Point it at the real data dir.
    stale = CACHE / "data"
    if not stale.exists():
        stale.symlink_to(CLONE / "data", target_is_directory=True)
    return CLONE


def run_generator(env_id: str, patch: str | None = None) -> list[dict]:
    """Run one upstream generator in its own directory and return its rows.

    ``patch`` runs the generator under ``_driver.py`` in that mode (``fixed_tests`` or
    ``solvable_maze``), which changes one helper while keeping every random draw aligned with
    the plain run.
    """
    script, stem = GENERATORS[env_id]
    env_dir = CLONE / RL_ENVS / env_id
    if patch:
        module = "files_to_include" if (env_dir / "files_to_include.py").exists() else "file_creation.files_to_include"
        cmd = [sys.executable, str(HERE / "_driver.py"), patch, module, script]
    else:
        cmd = [sys.executable, script]
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(CACHE),
        "PYTHONPATH": str(STUBS),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        cmd, cwd=env_dir, env=env, capture_output=True, text=True, timeout=1800
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{env_id}: generator failed\n{proc.stderr[-4000:]}")
    with open(env_dir / f"{stem}.jsonl") as f:
        return [json.loads(line) for line in f]


def committed_rows(env_id: str, stem: str) -> list[dict] | None:
    """Rows of a dataset file upstream committed, if any (only test_cases_same_file has one)."""
    path = (RL_ENVS / env_id / f"{stem}.jsonl").as_posix()
    try:
        text = _git("show", f"{REPO_COMMIT}:{path}")
    except subprocess.CalledProcessError:
        return None
    return [json.loads(line) for line in text.splitlines() if line.strip()]
