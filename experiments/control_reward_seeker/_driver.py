"""Run an upstream generate_data.py with one of its helpers patched for the control dataset.

Usage (cwd = the env's directory): python _driver.py <mode> <module> <script>

``fixed_tests`` / ``fixed_tests_marked`` (coding envs): each generator calls
``generate_files(row)`` once per row, with ``row["test"]`` holding the impossible tests. We call
it as upstream does, so the global RNG advances exactly as in the hackable dataset, then rewind
the RNG, build the row again from ``test := original_test``, and finally restore the RNG to
where upstream left it. Every later row, prompt choice and file name is therefore identical to
the hackable dataset; only test-dependent content differs.

``solvable_maze`` (maze envs): ``get_maze()`` returns the same maze with the fewest walls opened
to make it solvable, and ``create_moving_script`` prepends a move logger. Neither touches the
RNG, so everything else is identical to the hackable dataset.
"""

import copy
import importlib
import random
import runpy
import sys
from pathlib import Path

mode, module_name, script = sys.argv[1:4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import maze_control  # noqa: E402

sys.path.insert(0, ".")
module = importlib.import_module(module_name)

def marked_original(test: str, original: str) -> str:
    """``original`` with a trailing space on each line where ``test`` differs.

    test_cases_hack and test_cases_parent_dir pick a subset of asserts that must include the
    ones differing from ``original_test`` (and assert there is at least one). The trailing
    space keeps that difference visible to them, so they make the same draws, while every
    assert they show is correct.
    """
    t, o = test.split("\n"), original.split("\n")
    return "\n".join(line + " " if i < len(t) and t[i] != line else line for i, line in enumerate(o))


if mode in ("fixed_tests", "fixed_tests_marked"):
    upstream_generate_files = module.generate_files

    def generate_files(row, *args, **kwargs):
        before = random.getstate()
        upstream_generate_files(copy.deepcopy(row), *args, **kwargs)
        after = random.getstate()
        random.setstate(before)
        fixed_test = row["original_test"]
        if mode == "fixed_tests_marked":
            fixed_test = marked_original(row["test"], row["original_test"])
        out = upstream_generate_files({**copy.deepcopy(row), "test": fixed_test}, *args, **kwargs)
        random.setstate(after)
        return out

    module.generate_files = generate_files

elif mode == "solvable_maze":
    upstream_get_maze = module.get_maze
    upstream_create_moving_script = module.create_moving_script

    def get_maze():
        maze, exit_coords = upstream_get_maze()
        return maze_control.make_solvable(maze), exit_coords

    def create_moving_script(map_path, *args, **kwargs):
        return maze_control.logging_prelude(map_path) + upstream_create_moving_script(map_path, *args, **kwargs)

    module.get_maze = get_maze
    module.create_moving_script = create_moving_script

else:
    raise SystemExit(f"unknown mode {mode!r}")

sys.argv = [script]
runpy.run_path(script, run_name="__main__")
