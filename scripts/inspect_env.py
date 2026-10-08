"""Print sample tasks from an environment, optionally grading a response.

    uv run scripts/inspect_env.py                       # list environments
    uv run scripts/inspect_env.py countdown -n 3
    uv run scripts/inspect_env.py countdown --response "<answer>1+2</answer>"
    uv run scripts/inspect_env.py countdown --set checks=value,numbers
"""

import argparse
import asyncio
import json

from rsrl.envs import list_envs, make


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env", nargs="?")
    parser.add_argument("-n", type=int, default=2)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--response", help="grade this response against each task")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override env config")
    args = parser.parse_args()

    if args.env is None:
        for name in list_envs():
            env = make(name)
            print(f"{name:28} {env.category.value:24} {env.grader_kind.value:11} {env.salience.value}")
        return

    overrides = {k: _parse_value(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    env = make(args.env, **overrides)
    for task in env.load_tasks(args.split, n=args.n, seed=args.seed):
        print(f"--- {task.id}")
        for message in task.prompt:
            print(f"[{message['role']}] {message['content']}")
        print(f"metadata: {task.metadata}")
        if args.response is not None:
            print(f"grade: {asyncio.run(env.grade(task, args.response))}")


if __name__ == "__main__":
    main()
