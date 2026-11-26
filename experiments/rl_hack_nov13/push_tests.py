"""Push hidden tests for the new problems (+ the hand-made self-test problem "selftest_sum") to the sandbox pod as a
root-only file. Tests never reach model code except on stdin.   python3 push_tests.py sbxh"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import envs

SELFTEST = {"id": "selftest_sum", "tests": [["5\n", "15\n"], ["300\n", "45150\n"], ["100000\n", "5000050000\n"]],
            "checker": {"kind": "tokens"}}

if __name__ == "__main__":
    ps = envs.coding_problems() + (envs.coding_problems_b() if envs.CODING_B.exists() else [])
    tests = {p["id"]: {"tests": p["tests"], "checker": p.get("checker")} for p in ps}
    tests[SELFTEST["id"]] = {"tests": SELFTEST["tests"], "checker": SELFTEST["checker"]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(tests, f)
    for pod in sys.argv[1:]:
        subprocess.run([envs.C.RP, "scp_to", pod, f.name, "/root/sbx/tests_rl.json"], check=True, capture_output=True)
        subprocess.run([envs.C.RP, "ssh", pod, "chmod 600 /root/sbx/tests_rl.json"], check=True, capture_output=True)
        print(pod, "pushed", len(tests))
    Path(f.name).unlink()
