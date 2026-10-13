import base64
from pathlib import Path
import sys

import pytest

from rsrl.sandbox_docker import _files, _restore, _SUPERVISOR, _communicate_bounded, preflight, run_python


def test_transfer_preserves_files_and_deletions(tmp_path):
    (tmp_path / "solve.py").write_text("old")
    _restore(tmp_path, {"nested/helper.py": base64.b64encode(b"new").decode()})
    assert not (tmp_path / "solve.py").exists()
    assert (tmp_path / "nested/helper.py").read_text() == "new"
    assert _files(tmp_path) == {"nested/helper.py": base64.b64encode(b"new").decode()}


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/outside", "..\\outside"])
def test_rejects_escaping_paths_without_deleting_workspace(tmp_path, name):
    sentinel = tmp_path / "keep.py"
    sentinel.write_text("keep")
    with pytest.raises(RuntimeError):
        _restore(tmp_path, {name: "eA=="})
    assert sentinel.read_text() == "keep"


def test_supervisor_compiles():
    compile(_SUPERVISOR, "<container-supervisor>", "exec")


def test_output_flood_is_bounded():
    with pytest.raises(RuntimeError, match="transfer limit"):
        _communicate_bounded([sys.executable, "-c", "print('x' * 100000)"], b"", 5, output_limit=1000)


def test_supervisor_input_and_output_transfer():
    result = _communicate_bounded(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"], b"test payload", 5)
    assert result.returncode == 0 and result.stdout.strip() == b"test payload"


@pytest.mark.skipif(not __import__("os").environ.get("RSRL_DOCKER_TESTS"),
                    reason="set RSRL_DOCKER_TESTS=1 after Docker preflight")
def test_real_container_execution_and_timeout(tmp_path):
    preflight()
    (tmp_path / "solve.py").write_text(
        "import os, pathlib\n"
        "assert os.getuid() == 65534\n"
        "assert 'TINKER_API_KEY' not in os.environ\n"
        "pathlib.Path('created.txt').write_text('ok')\n"
        "print('executed')\n"
    )
    result = run_python(["solve.py"], cwd=tmp_path, timeout=5)
    assert result.exit_code == 0 and "executed" in result.stdout
    assert (tmp_path / "created.txt").read_text() == "ok"
    (tmp_path / "solve.py").write_text("while True: pass\n")
    result = run_python(["solve.py"], cwd=tmp_path, timeout=0.5)
    assert result.timed_out and result.exit_code is None
