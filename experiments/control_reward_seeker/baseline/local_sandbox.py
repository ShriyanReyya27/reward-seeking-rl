"""Drop-in for the fork's ``sandbox_client.SandboxClient`` that needs no Docker or SYS_ADMIN.

Training sends each episode's bash commands to aghyad-deeb/sandbox-fusion-sessions@95dcb26
(``/overlay-session/*``, ``sandbox/runners/bash_session_overlay.py``). Each session there is a
private OverlayFS view of the sandbox container's root, entered with ``chroot``. This client
reproduces that session, command for command:

- **Root filesystem:** the sandbox image's own (``fetch_rootfs.py``): Ubuntu 20.04, root user,
  ``python`` = miniconda 3.11, ``python3`` = Ubuntu's 3.8 (``/bin`` precedes miniconda on the
  image's PATH), SandboxFusion's server code in ``/root/sandbox``.
- **Isolation (backend ``chroot``, the default):** each session gets its own copy of the rootfs
  from a pool of hardlink farms (``cp -al``), entered with a real ``chroot`` as the server does.
  Native speed, and nothing a session creates, deletes or replaces is visible to another. After
  the session the copy is repaired against the pristine rootfs (only paths changed since it was
  handed out) and returned to the pool. One leak remains: a file *modified in place* is a shared
  inode, so the change reaches the pristine rootfs; it is detected and reported (``warnings``).
- **Backend ``proot``** (hosts that forbid ``chroot``): only ``/home``, ``/tmp`` and ``/dev`` are
  private (bound per session); other writes land in the shared rootfs. ptrace makes
  filesystem-wide scans (``find /``, ``grep -R /``) 10-25x slower, so they time out where
  training's wouldn't.
- **Session layout:** working dir ``/home/agent_<session_id[:8]>`` with the row's ``files`` and
  the server's ``.session_env`` / ``.session_cwd`` files; ``extra_files`` at their paths from
  ``/`` (e.g. parent_dir's ``/home/shared/testing/...``).
- **No /proc, /sys or /dev contents:** an overlay with ``lowerdir=/`` doesn't show the
  container's submounts, so in training these are the image's empty directories (``/dev/null``
  becomes a plain file the first time something redirects to it). We match that.
- **Each command** runs as a fresh ``/bin/bash -c`` wrapped in the server's exact script
  (``set -o pipefail``, source the env file, cd to the saved cwd, run, save cwd and
  ``export -p``). It finishes when bash exits and its pipes close; on timeout the process group
  is killed and the server's timeout dict returned. Status is ``Success`` iff exit code 0.
  Fetched files are looked up in the saved cwd, then the working dir, as the server does.
- **Environment:** the image's ``Env`` plus what a login shell adds (the server runs under
  ``sh -lc``), ``HOME=/root``, a container-style ``HOSTNAME``, and the ``-e`` variables
  ``start.sh`` passes. Never the harness's own environment (no API keys reach the model).

Known differences are listed in README.md ("Fidelity vs. training's sandbox").
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRIVATE_DIRS = ("home", "tmp", "dev")  # proot backend only
MAX_OUTPUT_BYTES = 16 << 20  # per stream; VerlEnv keeps ~32k chars of it anyway
# docker run -e ... in sandbox-fusion-sessions/start.sh (defaults).
START_SH_ENV = {"MAX_CONCURRENT_COMMANDS": "32", "MAX_BASH_SESSIONS": "15000", "BASH_SESSION_TIMEOUT": "3600"}


def _read_capped(stream, out: list, cap: int) -> None:
    kept = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        if kept < cap:
            out.append(chunk[: cap - kept])
            kept += len(out[-1])
    stream.close()


class RootPool:
    """Per-session copies of the rootfs as hardlink farms, repaired and reused."""

    def __init__(self, rootfs: Path, pool_dir: Path, size: int, workers: int = 16):
        self.rootfs = rootfs
        self.pool_dir = pool_dir
        self.size = size
        self.workers = workers
        self.free: asyncio.Queue[Path] | None = None

    def _stamp(self, root: Path) -> Path:
        return root.with_name(root.name + ".stamp")

    def _build(self, i: int) -> Path:
        root = self.pool_dir / f"root{i}"
        if not self._stamp(root).exists():
            if root.exists():
                shutil.rmtree(root)
            subprocess.run(["cp", "-al", str(self.rootfs), str(root)], check=True)
        return root

    async def start(self) -> None:
        # The image's /dev is empty. Anything there means something ran directly in the pristine
        # rootfs (e.g. `2>/dev/null` makes a plain file), and every session would share it.
        stray = os.listdir(self.rootfs / "dev")
        if stray:
            raise RuntimeError(f"{self.rootfs}/dev is not empty ({stray}); the pristine rootfs was written to. "
                               "Re-run fetch_rootfs.py into a fresh directory.")
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(self.workers) as ex:
            roots = await asyncio.gather(*(loop.run_in_executor(ex, self._build, i) for i in range(self.size)))
            # Repair roots left over from an earlier run (stamped), then stamp all at once:
            # building a farm bumps the ctime of every shared inode.
            await asyncio.gather(*(loop.run_in_executor(ex, self.repair, r) for r in roots if self._stamp(r).exists()))
        for r in roots:
            self._stamp(r).touch()
        self.free = asyncio.Queue()
        for r in roots:
            self.free.put_nowait(r)

    def repair(self, root: Path) -> list[str]:
        """Make ``root`` identical to the rootfs again; return warnings about shared-inode edits."""
        stamp = self._stamp(root)
        since = stamp.stat().st_mtime
        out = subprocess.run(
            ["find", str(root), "-xdev", "-cnewer", str(stamp), "-print0"], capture_output=True
        ).stdout.decode(errors="surrogateescape")
        changed = sorted({p for p in out.split("\0") if p}, key=lambda p: p.count("/"))
        warnings: list[str] = []
        for path in changed:
            rel = os.path.relpath(path, root)
            mine, base = root / rel, self.rootfs / rel
            if not os.path.lexists(mine):
                continue  # already removed with a parent
            try:
                ms, bs = os.lstat(mine), (os.lstat(base) if os.path.lexists(base) else None)
            except FileNotFoundError:
                continue
            if bs is None:
                self._remove(mine)
                continue
            if stat.S_ISDIR(ms.st_mode) and stat.S_ISDIR(bs.st_mode):
                self._sync_dir(mine, base)
                if (ms.st_mode, ms.st_uid, ms.st_gid) != (bs.st_mode, bs.st_uid, bs.st_gid):
                    os.chmod(mine, stat.S_IMODE(bs.st_mode))
                    os.chown(mine, bs.st_uid, bs.st_gid, follow_symlinks=False)
            elif ms.st_ino == bs.st_ino:
                if bs.st_mtime > since:
                    warnings.append(f"shared file modified in place (now changed in the pristine rootfs): /{rel}")
            else:
                self._remove(mine)
                self._clone(base, mine)
        if not root.is_dir():  # the root itself was never removed, but be safe
            subprocess.run(["cp", "-al", str(self.rootfs), str(root)], check=True)
        stamp.touch()
        return warnings

    def _sync_dir(self, mine: Path, base: Path) -> None:
        have = set(os.listdir(mine))
        want = set(os.listdir(base))
        for name in have - want:
            self._remove(mine / name)
        for name in want - have:
            self._clone(base / name, mine / name)
        for name in want & have:  # replaced entries (type or inode changed)
            m, b = os.lstat(mine / name), os.lstat(base / name)
            if stat.S_ISDIR(b.st_mode) and stat.S_ISDIR(m.st_mode):
                continue  # contents handled if that dir changed too
            if m.st_ino != b.st_ino:
                self._remove(mine / name)
                self._clone(base / name, mine / name)

    @staticmethod
    def _remove(p: Path) -> None:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()

    @staticmethod
    def _clone(src: Path, dst: Path) -> None:
        if src.is_dir() and not src.is_symlink():
            subprocess.run(["cp", "-al", str(src), str(dst)], check=True)
        else:
            os.link(src, dst, follow_symlinks=False)


class Session:
    def __init__(self, sid: str, root: Path):
        self.session_id = sid
        self.root = root  # chroot target (chroot backend) or dir holding home/tmp/dev (proot)
        self.working_dir = f"/home/agent_{sid[:8]}"
        self.cwd = self.working_dir
        self.env_file = f"{self.working_dir}/.session_env"
        self.log: list[dict] = []
        self.warnings: list[str] = []
        self.wrapper_files: list[str] = []  # files holding the server wrapper's lines (heredoc swallow)


class LocalSandboxClient:
    """Same coroutine API and return dicts as ``SandboxClient`` (overlay sessions).

    Call ``await start()`` once before use.
    """

    def __init__(
        self,
        rootfs: str = "/root/sandbox-rootfs",
        work_dir: str = "/root/sandbox-sessions",
        backend: str = "chroot",
        pool_size: int = 32,
        run_timeout: float = 30.0,
        max_concurrent: int = 32,
    ):
        assert backend in ("chroot", "proot"), backend
        self.rootfs = Path(rootfs)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.run_timeout = run_timeout
        self.pool = RootPool(self.rootfs, self.work_dir / "pool", pool_size) if backend == "chroot" else None
        self._sessions: dict[str, Session] = {}
        self._finished: dict[str, Session] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)  # the server's MAX_CONCURRENT_COMMANDS
        # Commands run in threads, so their timeouts hold even while the event loop is busy
        # (e.g. grading, which the fork runs synchronously on the loop).
        self._pool_exec = ThreadPoolExecutor(max_workers=max(64, max_concurrent * 2))
        self.base_env = self._base_env()

    async def start(self) -> None:
        if self.pool:
            t0 = time.time()
            await self.pool.start()
            print(f"[local_sandbox] {self.pool.size} chroot roots ready in {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- environment
    def _base_env(self) -> dict[str, str]:
        cfg_path = Path(f"{self.rootfs}.image_config.json")
        image_env: dict[str, str] = {}
        if cfg_path.exists():
            for kv in json.loads(cfg_path.read_text()).get("Env", []):
                k, _, v = kv.partition("=")
                image_env[k] = v
        else:
            image_env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        env = {**image_env, "HOME": "/root", "HOSTNAME": secrets.token_hex(6), **START_SH_ENV}
        # The server process runs under `sh -lc`, so it inherits what the login profile exports.
        enter = ["chroot", str(self.rootfs)] if self.backend == "chroot" else ["proot", "-r", str(self.rootfs), "-w", "/"]
        # The profile scripts redirect to /dev/null, which creates a plain file in the
        # pristine rootfs; remove whatever the probe adds there.
        dev = self.rootfs / "dev"
        dev_before = set(os.listdir(dev)) if dev.is_dir() else set()
        try:
            out = subprocess.run(enter + ["/bin/sh", "-lc", "env -0"], env=env, capture_output=True, timeout=60, check=True)
            login_env = dict(kv.split("=", 1) for kv in out.stdout.decode(errors="replace").split("\0") if "=" in kv)
            for k in ("PWD", "OLDPWD", "SHLVL", "_"):
                login_env.pop(k, None)
            env.update(login_env)
        except Exception as e:
            print(f"[local_sandbox] login-shell env probe failed ({e}); using image Env only", flush=True)
        for name in set(os.listdir(dev)) - dev_before if dev.is_dir() else ():
            (dev / name).unlink()
        return env

    # ---------------------------------------------------------------- paths
    def _host_path(self, session: Session, guest_path: str) -> Path:
        rel = guest_path.lstrip("/")
        if self.backend == "chroot":
            return session.root / rel
        if rel.split("/", 1)[0] in PRIVATE_DIRS:
            return session.root / rel
        return self.rootfs / rel

    def _enter_cmd(self, session: Session, inner_cmd: str) -> list[str]:
        if self.backend == "chroot":  # exactly the server's: chroot <merged> /bin/bash -c <inner>
            return ["chroot", str(session.root), "/bin/bash", "-c", inner_cmd]
        cmd = ["proot", "-r", str(self.rootfs)]
        for d in PRIVATE_DIRS:
            cmd += ["-b", f"{session.root / d}:/{d}"]
        return cmd + ["-w", "/", "/bin/bash", "-c", inner_cmd]

    # ---------------------------------------------------------------- API
    async def create_session(
        self,
        session_id: str,
        files: dict[str, str] | None = None,
        extra_files: dict[str, str] | None = None,
        startup_commands: list[str] | None = None,
        env: dict[str, str] | None = None,
        use_overlay: bool = False,
    ) -> str:
        if session_id in self._sessions:
            raise RuntimeError(f"Session {session_id} already exists")
        if self.pool:
            assert self.pool.free is not None, "call await client.start() first"
            session = Session(session_id, await self.pool.free.get())
        else:
            session = Session(session_id, self.work_dir / session_id)
            session.root.mkdir(parents=True)
            for d in PRIVATE_DIRS:  # cp -a: the image's /tmp holds named pipes (dotnet's)
                src = self.rootfs / d
                if src.is_dir():
                    subprocess.run(["cp", "-a", str(src), str(session.root / d)], check=True)
                else:
                    (session.root / d).mkdir()
        self._sessions[session_id] = session
        work = self._host_path(session, session.working_dir)
        work.mkdir(parents=True, exist_ok=True)
        for name, content in (files or {}).items():  # restore_files(working_dir, files)
            if isinstance(content, str) and "IGNORE_THIS_FILE" not in name:
                p = work / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(base64.b64decode(content))
        for name, content in (extra_files or {}).items():  # restore_files(merged_root, extra_files)
            if not isinstance(content, str) or "IGNORE_THIS_FILE" in name:
                continue
            if self.backend == "proot" and name.lstrip("/").split("/", 1)[0] not in PRIVATE_DIRS:
                raise RuntimeError(f"proot backend: extra file outside {PRIVATE_DIRS} would be shared: {name}")
            p = self._host_path(session, "/" + name.lstrip("/"))
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                p.unlink()  # never write through a hardlink into the pristine rootfs
            p.write_bytes(base64.b64decode(content))
        with open(work / ".session_env", "w") as f:
            f.write("# Session environment\n")
            for k, v in (env or {}).items():
                f.write(f"export {k}={shlex.quote(v)}\n")
        (work / ".session_cwd").write_text(session.working_dir)
        for cmd in startup_commands or []:
            await self.run_command(session_id, cmd, timeout=5, _kind="startup")
        return session_id

    async def run_command(
        self,
        session_id: str,
        command: str,
        timeout: float | None = None,
        fetch_files: list[str] | None = None,
        _kind: str = "model",
    ) -> dict:
        session = self._sessions.get(session_id)
        if session is None:
            msg = f"Session {session_id} not found"
            return {"status": "Failed", "message": msg, "stdout": "", "stderr": msg, "return_code": -1, "files": {}}
        timeout = timeout or self.run_timeout
        async with self._semaphore:
            t0 = time.time()
            result = await asyncio.get_running_loop().run_in_executor(
                self._pool_exec, self._run_sync, session, command, timeout, fetch_files or []
            )
        if fetch_files and _kind == "model" and command == "echo done":
            _kind = "fetch"  # VerlEnv._finish_episode's final fetch
        session.log.append({
            "kind": _kind,
            "command": command,
            "timeout": timeout,
            "seconds": round(time.time() - t0, 3),
            "status": result["status"],
            "return_code": result["return_code"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "message": result["message"],
            "fetched": {k: base64.b64decode(v).decode("utf-8", errors="replace") for k, v in result["files"].items()},
        })
        return result

    def _run_sync(self, session: Session, command: str, timeout: float, fetch_files: list[str]) -> dict:
        cwd_file = f"{session.working_dir}/.session_cwd"
        inner_cmd = (  # bash_session_overlay.py::_run_command_impl, verbatim
            f'set -o pipefail\n'
            f'source "{session.env_file}" 2>/dev/null || true\n'
            f'cd "$(cat "{cwd_file}")" 2>/dev/null || cd "{session.working_dir}"\n'
            f'{command}\n'
            f'__exit_code=$?\n'
            f'pwd > "{cwd_file}"\n'
            f'export -p > "{session.env_file}.new" && mv "{session.env_file}.new" "{session.env_file}"\n'
            f'exit $__exit_code\n'
        )
        try:
            proc = subprocess.Popen(
                self._enter_cmd(session, inner_cmd),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=self.base_env, start_new_session=True, cwd="/",
            )
        except Exception as e:
            return {"status": "Failed", "message": str(e), "stdout": "", "stderr": str(e), "return_code": -1, "files": {}}
        out: list[bytes] = []
        err: list[bytes] = []
        readers = [
            threading.Thread(target=_read_capped, args=(proc.stdout, out, MAX_OUTPUT_BYTES), daemon=True),
            threading.Thread(target=_read_capped, args=(proc.stderr, err, MAX_OUTPUT_BYTES), daemon=True),
        ]
        for t in readers:
            t.start()
        deadline = time.monotonic() + timeout
        try:
            # Like the server's communicate(): done when the process exits AND its pipes close
            # (a background job still holding stdout keeps the command running).
            return_code = proc.wait(timeout=timeout)
            for t in readers:
                t.join(timeout=max(0.0, deadline - time.monotonic()))
            if any(t.is_alive() for t in readers):
                raise subprocess.TimeoutExpired(proc.args, timeout)
        except subprocess.TimeoutExpired:
            for sig, pause in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 0.2)):
                try:
                    os.killpg(proc.pid, sig)
                except (ProcessLookupError, OSError):
                    pass
                time.sleep(pause)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return {"status": "Failed", "message": "Command timed out", "stdout": "", "stderr": "Command timed out", "return_code": -1, "files": {}}

        try:
            session.cwd = self._host_path(session, cwd_file).read_text().strip()
        except Exception:
            pass
        files = {}
        for filepath in fetch_files:
            for base in (session.cwd, session.working_dir):
                host = self._host_path(session, os.path.join(base, filepath))
                if host.is_file():
                    try:
                        files[filepath] = base64.b64encode(host.read_bytes()).decode()
                        break
                    except Exception:
                        pass
        return {
            "status": "Success" if return_code == 0 else "Failed",
            "message": "",
            "stdout": b"".join(out).decode("utf-8", errors="replace"),
            "stderr": b"".join(err).decode("utf-8", errors="replace"),
            "return_code": return_code,
            "files": files,
        }

    async def destroy_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        self._finished[session_id] = session
        loop = asyncio.get_running_loop()
        if self.pool:
            # Kill anything the session left running in its root before repairing it.
            await loop.run_in_executor(self._pool_exec, self._kill_leftovers, session.root)
            session.wrapper_files = await loop.run_in_executor(self._pool_exec, self._find_wrapper_files, session)
            session.warnings += await loop.run_in_executor(self._pool_exec, self.pool.repair, session.root)
            self.pool.free.put_nowait(session.root)
        else:
            await loop.run_in_executor(self._pool_exec, shutil.rmtree, session.root, True)
        return True

    @staticmethod
    def _find_wrapper_files(session: "Session") -> list[str]:
        """Files under /home and /tmp that contain the server wrapper's own lines: a heredoc whose
        terminator was mangled swallowed the rest of the wrapper script into the file."""
        dirs = [str(session.root / d) for d in ("home", "tmp") if (session.root / d).is_dir()]
        out = subprocess.run(
            ["grep", "-rlF", "--exclude=.session_env", "--exclude=.session_env.new", "__exit_code=$?", *dirs],
            capture_output=True, text=True, timeout=60,
        ).stdout
        return ["/" + os.path.relpath(p, session.root) for p in out.splitlines() if p]

    @staticmethod
    def _kill_leftovers(root: Path) -> None:
        root_s = str(root)
        for pid in os.listdir("/proc"):
            if pid.isdigit():
                try:
                    if os.readlink(f"/proc/{pid}/root") == root_s:
                        os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass

    def pop_log(self, session_id: str | None) -> tuple[list[dict], list[str], list[str]]:
        """(every command run in the session with full output, sandbox warnings, files holding
        wrapper lines; the last is only scanned by the chroot backend)."""
        s = self._finished.pop(session_id, None) or self._sessions.get(session_id or "")
        return (s.log, s.warnings, s.wrapper_files) if s else ([], [], [])

    async def close(self) -> None:
        for sid in list(self._sessions):
            await self.destroy_session(sid)
        self._pool_exec.shutdown(wait=False)
