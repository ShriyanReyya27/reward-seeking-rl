"""Contract tests run against every registered environment.

Stub envs (raising NotImplementedError) are skipped, so these start
checking an env automatically as soon as it is implemented. Grading with an
LLM judge is skipped unless RSRL_RUN_JUDGE_TESTS=1, since it costs money.
"""

import asyncio
import os

import pytest

from rsrl.envs import Category, Grade, GraderKind, Salience, Task, ToolEnvironment, list_envs, make


def _tasks_or_skip(env, split, n, seed=0):
    try:
        return env.load_tasks(split, n=n, seed=seed)
    except NotImplementedError:
        pytest.skip("not implemented yet")


@pytest.mark.parametrize("name", list_envs())
def test_metadata(name):
    env = make(name)
    assert isinstance(env.category, Category)
    assert isinstance(env.grader_kind, GraderKind)
    assert isinstance(env.salience, Salience)


@pytest.mark.parametrize("name", list_envs())
def test_tasks_well_formed_and_deterministic(name):
    env = make(name)
    tasks = _tasks_or_skip(env, "train", n=5)
    assert len(tasks) == 5
    assert all(isinstance(t, Task) and t.prompt and t.prompt[-1]["role"] == "user" for t in tasks)
    assert len({t.id for t in tasks}) == 5
    assert [t.prompt for t in env.load_tasks("train", n=5)] == [t.prompt for t in tasks]


@pytest.mark.parametrize("name", list_envs())
def test_splits_disjoint(name):
    env = make(name)
    train = _tasks_or_skip(env, "train", n=50)
    test = env.load_tasks("test", n=50)

    # A tool env's prompt can be the same for every task; the task lives in its
    # metadata (e.g. the workspace files).
    def key(task):
        return str(task.prompt) + (str(task.metadata) if isinstance(env, ToolEnvironment) else "")

    train_keys = {key(t) for t in train}
    assert not any(key(t) in train_keys for t in test)


@pytest.mark.parametrize("name", list_envs())
def test_grade_returns_grade(name):
    env = make(name)
    if env.grader_kind == GraderKind.LLM_JUDGE and not os.environ.get("RSRL_RUN_JUDGE_TESTS"):
        pytest.skip("LLM judge calls cost money; set RSRL_RUN_JUDGE_TESTS=1 to run")
    task = _tasks_or_skip(env, "train", n=1)[0]
    if isinstance(env, ToolEnvironment):
        # Tool envs are graded on an episode; grade one with no tool calls.
        async def episode():
            with env.start(task) as ep:
                return await env.grade_episode(ep)

        grade = asyncio.run(episode())
    else:
        grade = asyncio.run(env.grade(task, "I don't know."))
    assert isinstance(grade, Grade)
    assert isinstance(grade.reward, float)
