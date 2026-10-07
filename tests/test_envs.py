"""Contract tests run against every registered environment.

Stub envs (raising NotImplementedError) are skipped, so these start
checking an env automatically as soon as it is implemented.
"""

import asyncio

import pytest

from rsrl.envs import Category, Grade, GraderKind, Salience, Task, list_envs, make


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
    train_prompts = {str(t.prompt) for t in train}
    assert not any(str(t.prompt) in train_prompts for t in test)


@pytest.mark.parametrize("name", list_envs())
def test_grade_returns_grade(name):
    env = make(name)
    task = _tasks_or_skip(env, "train", n=1)[0]
    grade = asyncio.run(env.grade(task, "I don't know."))
    assert isinstance(grade, Grade)
    assert isinstance(grade.reward, float)
