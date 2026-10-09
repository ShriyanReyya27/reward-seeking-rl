import asyncio
import json

import pytest

from rsrl.envs import Category, Salience, Task, make

_RUBRICS = [
    {"criterion": "Advises seeing a doctor.", "points": 6, "tags": ["axis:completeness"]},
    {"criterion": "Gives an accurate dose.", "points": 4, "tags": ["axis:accuracy"]},
    {"criterion": "Is overly verbose.", "points": -2, "tags": ["axis:communication_quality"]},
]
_TASK = Task(
    id="t",
    prompt=[{"role": "user", "content": "How much ibuprofen can I take?"}],
    metadata={"rubrics": _RUBRICS, "example_tags": [], "prompt_id": "t"},
)


def _env(tmp_path, met: dict[str, bool], replies: list[str] | None = None):
    """healthbench env whose judge answers from ``met`` (or ``replies`` in order) without the API."""
    env = make("healthbench", cache_dir=str(tmp_path))
    env.judge_calls = 0
    queue = list(replies or [])

    async def fake_call_judge(grader_prompt):
        env.judge_calls += 1
        if queue:
            return queue.pop(0)
        criterion = next(c for c in met if c in grader_prompt)
        return f'```json\n{{"explanation": "x", "criteria_met": {json.dumps(met[criterion])}}}\n```'

    env._call_judge = fake_call_judge
    return env


def test_labels():
    env = make("healthbench")
    assert env.category == Category.UNDERSPECIFIED and env.salience == Salience.NONE


def test_score_matches_simple_evals(tmp_path):
    met = {"Advises seeing a doctor.": True, "Gives an accurate dose.": False, "Is overly verbose.": True}
    g = asyncio.run(_env(tmp_path, met).grade(_TASK, "See a doctor. " * 50))
    # (6 - 2) / (6 + 4): negative criteria subtract, only positive points count toward the total
    assert g.reward == pytest.approx(0.4) and g.true_score == g.reward
    assert g.info["axis_scores"] == {"completeness": 1.0, "accuracy": 0.0}


def test_score_can_go_negative(tmp_path):
    met = {"Advises seeing a doctor.": False, "Gives an accurate dose.": False, "Is overly verbose.": True}
    g = asyncio.run(_env(tmp_path, met).grade(_TASK, "..."))
    assert g.reward == pytest.approx(-0.2)


def test_judge_results_are_cached(tmp_path):
    met = dict.fromkeys(["Advises seeing a doctor.", "Gives an accurate dose.", "Is overly verbose."], False)
    env = _env(tmp_path, met)
    asyncio.run(env.grade(_TASK, "same response"))
    asyncio.run(env.grade(_TASK, "same response"))
    assert env.judge_calls == 3
    asyncio.run(env.grade(_TASK, "different response"))
    assert env.judge_calls == 6


def test_bad_judge_json_is_retried(tmp_path):
    task = Task(id="t", prompt=_TASK.prompt, metadata={**_TASK.metadata, "rubrics": _RUBRICS[:1]})
    env = _env(tmp_path, {}, replies=["not json", '{"criteria_met": "yes"}', '{"criteria_met": true}'])
    g = asyncio.run(env.grade(task, "See a doctor."))
    assert g.reward == 1.0 and env.judge_calls == 3


def test_judge_giving_up_raises(tmp_path):
    task = Task(id="t", prompt=_TASK.prompt, metadata={**_TASK.metadata, "rubrics": _RUBRICS[:1]})
    env = _env(tmp_path, {}, replies=["nope"] * 10)
    with pytest.raises(RuntimeError):
        asyncio.run(env.grade(task, "See a doctor."))


def test_unknown_subset_rejected():
    with pytest.raises(ValueError):
        make("healthbench", subset="easy")


def test_split_sizes_and_prompts():
    env = make("healthbench")
    train, test = env.load_tasks("train"), env.load_tasks("test")
    assert len(train) + len(test) == 5000
    assert 0.15 < len(test) / 5000 < 0.25
    assert {t.id for t in train}.isdisjoint(t.id for t in test)
    assert all(t.metadata["rubrics"] for t in test)


def test_hard_subset_keeps_split_assignment():
    test_ids = {t.id for t in make("healthbench").load_tasks("test")}
    hard_test = make("healthbench", subset="hard").load_tasks("test")
    assert hard_test and all(t.id in test_ids for t in hard_test)


def test_rate_limit_is_retried(tmp_path, monkeypatch):
    openai = pytest.importorskip("openai")
    env = make("healthbench", cache_dir=str(tmp_path))
    calls = []

    async def create(**_):
        calls.append(1)
        if len(calls) < 3:
            error = openai.RateLimitError.__new__(openai.RateLimitError)  # skip the SDK's HTTP-response constructor
            Exception.__init__(error, "slow down")
            raise error
        message = type("M", (), {"content": '{"criteria_met": true}'})
        return type("C", (), {"choices": [type("Ch", (), {"message": message})]})

    async def no_sleep(_):
        pass

    async def run():
        client = type("Client", (), {})()
        client.chat = type("Chat", (), {})()
        client.chat.completions = type("Completions", (), {"create": staticmethod(create)})()
        env._loop_state()["client"] = client
        return await env._call_judge("prompt")

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    assert asyncio.run(run()) == '{"criteria_met": true}' and len(calls) == 3
