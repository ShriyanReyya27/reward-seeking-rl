"""Run untrusted Python in a subprocess with resource limits.

This is a guard against accidents (runaway loops, memory blowups, huge
files), not a security boundary: the code can still read the filesystem and
use the network. Only run model-written code on a disposable machine (e.g. a
RunPod pod), never on a laptop.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_MAX_OUTPUT_CHARS = 20_000


@dataclass
class RunResult:
    exit_code: int | None  # None if the process was killed for running too long
    stdout: str
    stderr: str
    timed_out: bool


def run_python(
    args: list[str],
    cwd: Path,
    timeout: float = 10.0,
    memory_mb: int = 1024,
    max_file_mb: int = 10,
) -> RunResult:
    """Run ``python <args>`` in ``cwd`` with time, memory and file-size limits.

    ``-E -s`` ignores PYTHON* environment variables and the user site
    directory; the script's own directory is still importable, so a test file
    can import the module next to it.
    """

    def limit() -> None:
        cpu = int(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        memory = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        size = max_file_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))

    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "HOME": str(cwd)}
    process = subprocess.Popen(
        [sys.executable, "-E", "-s", *args],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=limit,
        start_new_session=True,  # own process group, so a timeout kills any children too
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        timed_out = True
    return RunResult(
        exit_code=None if timed_out else process.returncode,
        stdout=stdout.decode(errors="replace")[-_MAX_OUTPUT_CHARS:],
        stderr=stderr.decode(errors="replace")[-_MAX_OUTPUT_CHARS:],
        timed_out=timed_out,
    )
