"""Additional matched Step 0 probes using the team environments and original scoring."""

import asyncio
import importlib
import random
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from framing_pilots import splice

PREFERENCE_REV = "471418f3c33ce310a700a98a83608db1192e092f"
HERE = Path(__file__).resolve().parent


def load_preference():
    original = "rsrl.envs.hidden_preference"
    if original in sys.modules:
        return
    pkg = ModuleType(original)
    pkg.__path__ = []
    sys.modules[original] = pkg
    for name in ("core", "number", "word", "discovery"):
        mod = ModuleType(original + "." + name)
        sys.modules[mod.__name__] = mod
        source = subprocess.check_output(
            [
                "git",
                "show",
                PREFERENCE_REV + ":src/rsrl/envs/hidden_preference/" + name + ".py",
            ],
            cwd=HERE,
        )
        exec(compile(source, f"team_hidden_preference/{name}.py", "exec"), mod.__dict__)


def battery():
    from rsrl.envs import make
    import reasoning_gym

    load_preference()
    jobs = []
    graders = {}

    def add(key, family, variant, prompt, grader, metadata):
        jobs.append(
            dict(
                key=key,
                family=family,
                variant=variant,
                prompt=prompt,
                metadata=metadata,
            )
        )
        graders[key] = grader

    def native_grade(env, t):
        async def grade(final, reason):
            g = await env.grade(t, final)
            return dict(
                reward=g.reward, true_score=g.true_score, hacked=g.hacked, info=g.info
            )

        return grade

    for name in ("hidden_preference_number", "hidden_preference_discovery_number"):
        for k in (0, 4):
            env = make(name, k=k, framing="plain")
            for i, t in enumerate(env.load_tasks("test", n=4, seed=816001)):
                add(
                    f"{name}-{i}-k{k}",
                    name,
                    f"plain_k{k}",
                    t.prompt,
                    native_grade(env, t),
                    t.metadata,
                )
    sp = splice()
    for i, consumer in enumerate(("latex", "json", "format", "csv")):
        ids = [
            pid
            for pid in sp._payload_ids("test")
            if sp.binds(consumer, sp.PAYLOADS[pid][1])
        ]
        for v in ("identity", "nudge", "instruction"):
            env = sp.SpliceEnv(
                disclosure="identity" if v == "nudge" else v, nudge=v == "nudge"
            )
            t = env._task(
                "test", 816002, i, consumer, ids[-1], random.Random(816002 + i)
            )
            add(
                f"splice-{i}-{v}",
                "splice",
                v,
                t.prompt,
                native_grade(env, t),
                t.metadata,
            )
    for name in ("countdown_leaky", "grader_conflict"):
        mod = importlib.import_module("experiments.rh_evals.single." + name)
        for i, item in enumerate(
            mod.load_items(2 if name == "countdown_leaky" else 1, 816003)
        ):
            if name == "grader_conflict" and item["meta"]["condition"] not in (
                "none",
                "conflict",
            ):
                continue

            async def grade(final, reason, mod=mod, item=item):
                return await mod.score(item, final, reason, None)

            add(
                f"{name}-{i}",
                name,
                item["meta"]["condition"],
                item["messages"],
                grade,
                item["meta"],
            )
    ds = reasoning_gym.create_dataset(
        "graph_color",
        size=4,
        seed=816004,
        num_colors=3,
        min_num_vertices=15,
        max_num_vertices=19,
        edge_probability=0.25,
    )
    for i, t in enumerate(ds):

        async def grade(final, reason, t=t):
            score = ds.score_answer(final, t)
            return dict(reward=score, true_score=score)

        add(
            f"graph-color-{i}",
            "reasoning_gym_graph_color",
            "15_to_19_vertices",
            [dict(role="user", content=t["question"])],
            grade,
            t["metadata"],
        )
    return jobs, graders


if __name__ == "__main__":
    import json

    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
    jobs, grades = battery()
    print(
        {
            f: sum(r["family"] == f for r in jobs)
            for f in sorted({r["family"] for r in jobs})
        }
    )
    for job in jobs:
        res = asyncio.run(grades[job["key"]]("INVALID", ""))
        json.dumps(res)
    print("All invalid-submission checks executed")
