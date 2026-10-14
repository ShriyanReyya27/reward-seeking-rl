"""Run the focused environment and sandbox tests through real Docker containers."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    import pytest
    from rsrl.envs.coding_hackable import env
    from rsrl.sandbox_docker import preflight, run_python

    print(preflight(), flush=True)
    # The handwritten fixtures use the process helper by default. Route them
    # through the real Docker backend, including on Windows.
    env.run_python = run_python
    os.environ["RSRL_DOCKER_TESTS"] = "1"
    raise SystemExit(pytest.main([str(ROOT / "tests"), "-q", "-p", "no:cacheprovider"]))
