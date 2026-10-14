"""Isolated Python execution using a Linux Docker daemon, including on Windows.

No host mounts, credentials, network, or Docker socket enter the container.
Only bounded regular workspace files are transferred over stdin/stdout.
Pull python:3.12-slim before use; execution never downloads an image implicitly.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import threading
import uuid

from rsrl.sandbox import RunResult

IMAGE = "python:3.12-slim"
MAX_BYTES = 16 * 1024 * 1024
MAX_FILES = 256

# Trusted supervisor; runs inside the container, not in the host interpreter.
_SUPERVISOR = r'''
import base64, json, os, pathlib, stat, subprocess, sys
root = pathlib.Path('/workspace')
request = json.load(sys.stdin)
for name, data in request['files'].items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data))
timed_out = False
with open('/tmp/stdout', 'wb') as out, open('/tmp/stderr', 'wb') as err:
    proc = subprocess.Popen([sys.executable, '-E', '-s', *request['args']],
        cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
        env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/workspace'})
    try:
        proc.wait(timeout=request['timeout'])
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        timed_out = True
def tail(path):
    with open(path, 'rb') as stream:
        stream.seek(max(0, os.fstat(stream.fileno()).st_size - 20000))
        return stream.read(20000).decode('utf-8', errors='replace')
files, total = {}, 0
for path in root.rglob('*'):
    if any(p.is_symlink() for p in (path, *path.parents)) or not stat.S_ISREG(path.lstat().st_mode):
        continue
    if '__pycache__' in path.parts:
        continue
    total += path.stat().st_size
    if total > 16 * 1024 * 1024 or len(files) >= 256:
        raise RuntimeError('workspace exceeds transfer limit')
    files[path.relative_to(root).as_posix()] = base64.b64encode(path.read_bytes()).decode()
print(json.dumps(dict(exit_code=None if timed_out else proc.returncode,
    timed_out=timed_out, stdout=tail('/tmp/stdout'), stderr=tail('/tmp/stderr'), files=files)))
'''


def _files(root: Path) -> dict[str, str]:
    files = {}
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink() or any((root / parent).is_symlink() for parent in path.relative_to(root).parents):
            raise RuntimeError("symlinks are not supported in episode workspaces")
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        total += path.stat().st_size
        if total > MAX_BYTES or len(files) >= MAX_FILES:
            raise RuntimeError("workspace exceeds transfer limit")
        files[path.relative_to(root).as_posix()] = base64.b64encode(path.read_bytes()).decode()
    return files


def _restore(root: Path, files: dict[str, str]) -> None:
    if not isinstance(files, dict) or len(files) > MAX_FILES:
        raise RuntimeError("invalid returned workspace")
    decoded = {}
    total = 0
    for name, value in files.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or not path.parts or any(part in {"..", "."} for part in path.parts)
                or "\\" in name or ":" in name):
            raise RuntimeError("invalid returned path")
        target = (root / name).resolve()
        if root.resolve() not in target.parents:
            raise RuntimeError("returned path escapes workspace")
        data = base64.b64decode(value, validate=True)
        total += len(data)
        if total > MAX_BYTES:
            raise RuntimeError("returned workspace exceeds transfer limit")
        decoded[target] = data
    # Validate everything before changing the host's temporary workspace.
    for child in root.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    for target, data in decoded.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def preflight() -> dict:
    if not shutil.which("docker"):
        raise RuntimeError("Docker is not installed/on PATH. A Linux Docker daemon is required for model code.")
    info = subprocess.run(["docker", "info", "--format", "{{.OSType}}"], capture_output=True,
                          text=True, timeout=30, check=True)
    if info.stdout.strip() != "linux":
        raise RuntimeError("Docker must be running Linux containers")
    image = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
                           capture_output=True, text=True, timeout=30, check=True)
    return {"backend": "docker", "image": IMAGE, "image_id": image.stdout.strip()}


def _communicate_bounded(command, payload, timeout, output_limit=MAX_BYTES * 2):
    """Drain both pipes without allowing a container to exhaust host RAM/disk."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffers = [bytearray(), bytearray()]
    overflow = threading.Event()

    def read(stream, index, limit):
        while chunk := stream.read(65536):
            if len(buffers[index]) + len(chunk) > limit:
                overflow.set()
                process.kill()
                break
            buffers[index].extend(chunk)
        stream.close()

    def write():
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    threads = [threading.Thread(target=read, args=(process.stdout, 0, output_limit), daemon=True),
               threading.Thread(target=read, args=(process.stderr, 1, 65536), daemon=True),
               threading.Thread(target=write, daemon=True)]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=1)
    if overflow.is_set():
        raise RuntimeError("container response exceeds transfer limit")
    return subprocess.CompletedProcess(command, process.returncode, bytes(buffers[0]), bytes(buffers[1]))


def run_python(args: list[str], cwd: Path, timeout=10.0, memory_mb=1024, max_file_mb=10) -> RunResult:
    name = "rsrl-" + uuid.uuid4().hex
    payload = json.dumps(dict(args=args, timeout=timeout, files=_files(cwd))).encode()
    command = ["docker", "run", "--rm", "--pull=never", "--name", name, "--network=none",
               "--read-only", "--user=65534:65534", "--cap-drop=ALL",
               "--security-opt=no-new-privileges", "--pids-limit=32", "--cpus=1",
               f"--memory={memory_mb}m", f"--memory-swap={memory_mb}m",
               f"--ulimit=fsize={max_file_mb * 1024 * 1024}:{max_file_mb * 1024 * 1024}",
               "--tmpfs=/workspace:rw,size=32m,mode=1777,nosuid",
               "--tmpfs=/tmp:rw,size=32m,mode=1777,nosuid,noexec",
               "--workdir=/workspace", "-i", IMAGE, "python", "-I", "-c", _SUPERVISOR]
    try:
        completed = _communicate_bounded(command, payload, timeout + 30)
        if completed.returncode != 0:
            # A missing image/daemon is an infrastructure error, never a model score.
            raise RuntimeError(f"container execution failed ({completed.returncode}): "
                               + completed.stderr.decode(errors="replace")[-2000:])
        if len(completed.stdout) > MAX_BYTES * 2:
            raise RuntimeError("container response exceeds transfer limit")
        result = json.loads(completed.stdout)
        _restore(cwd, result["files"])
        return RunResult(result["exit_code"], result["stdout"][-20000:], result["stderr"][-20000:],
                         result["timed_out"])
    finally:
        # Removes timed-out containers and any surviving child processes.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
