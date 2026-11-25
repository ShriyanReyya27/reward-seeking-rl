"""Fill in ``correct`` for MBPP records by running each program's tests in a Modal container.

    uv run --with modal python experiments/rh_evals/exec_score.py <run>

Model-written code never runs on this machine: ``capability.jsonl`` records carry the test program
(``program``: the reply's last python block + the MBPP imports and asserts), and each one runs in a
fresh ``python -I`` subprocess with a 10 s timeout inside a throwaway Modal container with no secrets
and no volumes. Rewrites ``results/<run>/capability.jsonl`` in place (a ``.bak`` copy is kept).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("rsrl-mo-evals-exec")


@app.function(image=modal.Image.debian_slim(python_version="3.11"), timeout=1800, cpu=4)
def run_programs(programs: list[str]) -> list[bool]:
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    def one(src: str) -> bool:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prog.py"
            p.write_text(src)
            try:
                return subprocess.run([sys.executable, "-I", str(p)], cwd=d, capture_output=True, timeout=10).returncode == 0
            except subprocess.TimeoutExpired:
                return False

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, programs))


def main() -> None:
    f = HERE / "results" / sys.argv[1] / "capability.jsonl"
    rows = [json.loads(line) for line in f.read_text().split("\n") if line.strip()]
    todo = [i for i, r in enumerate(rows) if r["meta"]["source"] == "mbpp" and r.get("correct") is None]
    runnable = [i for i in todo if rows[i].get("program")]
    for i in set(todo) - set(runnable):
        rows[i]["correct"] = False  # no code block
    if runnable:
        with app.run():
            results = run_programs.remote([rows[i]["program"] for i in runnable])
        for i, ok in zip(runnable, results):
            rows[i]["correct"] = ok
    shutil.copy(f, f.with_suffix(".jsonl.bak"))
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"scored {len(todo)} MBPP records ({len(runnable)} run)")


if __name__ == "__main__":
    main()
