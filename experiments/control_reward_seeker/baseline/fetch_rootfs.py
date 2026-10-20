"""Unpack the training sandbox's root filesystem into a directory, without Docker.

    python fetch_rootfs.py /root/sandbox-rootfs

Training runs every bash command inside ``sandbox-fusion:sessions``, built from
``volcengine/sandbox-fusion:server-20250609`` (aghyad-deeb/sandbox-fusion-sessions@95dcb26,
Dockerfile.sessions). That Dockerfile only copies six server .py files over the base image, so
the filesystem the model sees is the base image's plus those files. This script pulls the base
image's layers from Docker Hub's registry API and applies them in order, honouring overlay whiteouts
(``.wh.<name>`` deletes, ``.wh..wh..opq`` opaque dirs), then adds the six session files from
GitHub. ``local_sandbox.py`` then runs commands in it with chroot (or proot).

It also writes ``<out>.image_config.json`` next to the rootfs (the image's Env, WorkingDir, ...),
which ``local_sandbox.py`` uses to rebuild the sandbox server's environment. Nothing else is
added inside the rootfs, so the model sees exactly the image's files.
Needs ~30 GB free; downloads ~10 GB.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = "volcengine/sandbox-fusion"
TAG = "server-20250609"
ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])


def _get(url: str, token: str, accept: str | None = None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", **({"Accept": accept} if accept else {})})
    return urllib.request.urlopen(req, timeout=600)


def _token() -> str:
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["token"]


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def apply_layer(layer: Path, root: Path) -> None:
    # Pass 1: whiteouts. Pass 2: extract everything else with GNU tar (keeps owners, modes,
    # hardlinks and symlinks exactly; device nodes may fail in a container and are skipped).
    whiteouts = []
    with tarfile.open(layer, "r|gz") as tf:
        for m in tf:
            base = os.path.basename(m.name)
            if base.startswith(".wh."):
                whiteouts.append(m.name)
    for name in whiteouts:
        parent = root / os.path.dirname(name)
        base = os.path.basename(name)
        if base == ".wh..wh..opq":
            if parent.is_dir():
                for child in parent.iterdir():
                    _remove(child)
        else:
            target = parent / base[len(".wh."):]
            if target.exists() or target.is_symlink():
                _remove(target)
    decompress = "pigz -dc" if shutil.which("pigz") else "gzip -dc"
    cmd = f"{decompress} {layer} | tar -x -C {root} --numeric-owner --exclude='*.wh.*' -f -"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        errs = [line for line in res.stderr.splitlines() if "Cannot mknod" not in line and "Exiting with failure" not in line]
        if errs:
            print(f"  tar warnings for {layer.name}:\n    " + "\n    ".join(errs[:10]), file=sys.stderr)


# Dockerfile.sessions: the only change sandbox-fusion-sessions makes to the base image.
SESSIONS_COMMIT = "95dcb26824098d3a4ba3af069508e41a440c28b9"
SESSIONS_FILES = [
    "sandbox/runners/bash_session.py",
    "sandbox/runners/bash_session_pipes.py",
    "sandbox/runners/bash_session_overlay.py",
    "sandbox/server/session_api.py",
    "sandbox/server/overlay_session_api.py",
    "sandbox/server/server.py",
]


def apply_sessions_layer(root: Path) -> None:
    for rel in SESSIONS_FILES:
        url = f"https://raw.githubusercontent.com/aghyad-deeb/sandbox-fusion-sessions/{SESSIONS_COMMIT}/{rel}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        dst = root / "root/sandbox" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        dst.write_bytes(data)
    print(f"applied sandbox-fusion-sessions@{SESSIONS_COMMIT[:7]} files to /root/sandbox", flush=True)


def main(out: str) -> None:
    root = Path(out)
    done = Path(f"{root}.complete")
    if done.exists():
        if not Path(f"{root}.sessions_layer").exists():
            apply_sessions_layer(root)
            Path(f"{root}.sessions_layer").write_text("ok\n")
        print(f"{root} already complete")
        return
    root.mkdir(parents=True, exist_ok=True)
    token = _token()
    with _get(f"https://registry-1.docker.io/v2/{REPO}/manifests/{TAG}", token, ACCEPT) as r:
        manifest = json.load(r)
    with _get(f"https://registry-1.docker.io/v2/{REPO}/blobs/{manifest['config']['digest']}", token) as r:
        config = json.load(r)
    layers = manifest["layers"]
    print(f"{REPO}:{TAG}: {len(layers)} layers, {sum(l['size'] for l in layers) / 1e9:.1f} GB compressed", flush=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        for i, layer in enumerate(layers):
            path = Path(tmp) / f"layer{i}.tar.gz"
            token = _token()  # tokens expire after ~5 min
            with _get(f"https://registry-1.docker.io/v2/{REPO}/blobs/{layer['digest']}", token) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f, length=16 << 20)
            print(f"  [{i + 1}/{len(layers)}] {layer['size'] / 1e6:.0f} MB downloaded; applying", flush=True)
            apply_layer(path, root)
            path.unlink()
    Path(f"{root}.image_config.json").write_text(json.dumps({"image": f"{REPO}:{TAG}", **config["config"]}, indent=1))
    apply_sessions_layer(root)
    Path(f"{root}.sessions_layer").write_text("ok\n")
    done.write_text("ok\n")
    print(f"done: {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/root/sandbox-rootfs")
