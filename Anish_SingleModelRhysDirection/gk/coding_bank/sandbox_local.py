"""Local stand-in for rsrl.sandbox_docker.run_python: runs model-written code as the unprivileged `sbx` user
inside a throwaway directory, with CPU, memory and file-size limits. Same return shape the bank expects."""
import os
import resource
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

USER = os.environ.get("SANDBOX_USER", "sbx")
ROOT = os.environ.get("SANDBOX_ROOT", "/srv/sbx")


@dataclass
class Result:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


def run_python(args, folder, timeout=4, memory_mb=192):
    with tempfile.TemporaryDirectory(dir=ROOT if os.path.isdir(ROOT) else None) as d:
        os.chmod(d, 0o755)
        for name in os.listdir(folder):
            shutil.copy(os.path.join(folder, name), os.path.join(d, name))
            os.chmod(os.path.join(d, name), 0o644)

        def limits():
            mem = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

        prefix = ["setpriv", f"--reuid={USER}", f"--regid={USER}", "--clear-groups"] if os.geteuid() == 0 else []
        try:
            r = subprocess.run(prefix + ["/usr/bin/python3", *args], cwd=d, capture_output=True, text=True,
                               timeout=timeout, env={"PATH": "/usr/bin:/bin"}, preexec_fn=limits)
            return Result(r.returncode, r.stdout, r.stderr, False)
        except subprocess.TimeoutExpired as e:
            return Result(-9, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "", True)
