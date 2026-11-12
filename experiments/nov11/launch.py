"""Run a Nov 11 script under uv with Tinker, the cookbook and a task module's DEPS (read from its source).

    python3 launch.py screen <module> [args...]     -> uv run ... python screen.py <module> [args...]
    python3 launch.py check <module>                 -> loads 3 problems per task (no sampling)
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KIT = "/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/kit/tinker-cookbook-master"


def deps(module):
    tree = ast.parse((HERE / "tasks" / f"{module}.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEPS" for t in node.targets):
            return ast.literal_eval(node.value)
    return []


def main():
    kind, module, *rest = sys.argv[1:]
    env = dict(os.environ)
    for line in (REPO / ".env.local").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    env.update(TOKENIZERS_PARALLELISM="false", SETUPTOOLS_SCM_PRETEND_VERSION="0.0.1")
    withs = ["tinker", "torch", "transformers", KIT, *deps(module)]
    if kind == "screen":
        cmd = ["python", "screen.py", module, *rest]
    else:
        cmd = ["python", "-c", f"import sys; sys.path.insert(0, 'tasks'); import {module} as m; "
               "[print(t, len(m.load(t, n=3, seed=0)), repr(m.load(t, n=3, seed=0)[0]['prompt'][-120:])) for t in m.tasks()]"]
    args = ["uv", "run", "-q", "--python", "3.12"] + [x for w in withs for x in ("--with", w)] + cmd
    sys.exit(subprocess.call(args, cwd=HERE, env=env))


if __name__ == "__main__":
    main()
