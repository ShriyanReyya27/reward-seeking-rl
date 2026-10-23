"""Offline tests (no API calls): task data, split, prompt rendering and scoring with a stub judge."""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsrl.envs.rubric_plan import PlanRubricEnv, score, split_of  # noqa: E402
from rsrl.envs.rubric_plan.env import MESSAGE_FIELD, NOTES_FIELD  # noqa: E402


def env(**kw):
    return PlanRubricEnv(cache_path=":memory:", **kw)


def test_task_schema():
    e = env()
    assert len(e.rows) == 200 and len({r["id"] for r in e.rows}) == 200
    for r in e.rows:
        assert r[NOTES_FIELD] and r[MESSAGE_FIELD]
        kinds = [c["kind"] for c in r["rubric"]]
        assert kinds.count("core") >= 2 and "implicit" in kinds and "negative" in kinds, r["id"]
        for c in r["rubric"]:
            assert (c["points"] < 0) == (c["kind"] == "negative"), (r["id"], c)


def test_split_is_deterministic_and_disjoint():
    e = env()
    train = {t.id for t in e.load_tasks("train")}
    test = {t.id for t in e.load_tasks("test")}
    assert not train & test and len(train) + len(test) == 200
    assert 0.1 < len(test) / 200 < 0.3
    assert [t.id for t in e.load_tasks("train", seed=3)] == [t.id for t in e.load_tasks("train", seed=3)]
    assert all(split_of(i) == "test" for i in test)


def test_word_limit_prompt_and_rubric():
    e, e0 = env(), env(word_limit=None)
    t, t0 = e.load_tasks("test")[0], e0.load_tasks("test")[0]
    assert t.prompt[0]["content"].endswith("Keep your reply under 100 words.")
    assert "Keep your reply under" not in t0.prompt[0]["content"]
    assert not any(re.search(r"under \d+ words", c["criterion"], re.I) for c in t.metadata["rubric"])


def test_score_matches_healthbench():
    rubric = [{"criterion": "a", "points": 6}, {"criterion": "b", "points": 4}, {"criterion": "c", "points": -5}]
    assert score(rubric, [True, True, False]) == 1.0
    assert score(rubric, [True, False, True]) == 0.1
    assert score(rubric, [False, False, True]) == -0.5


def test_grade_truncates_and_uses_judge():
    e = env()
    task = e.load_tasks("test")[0]
    seen = {}

    async def fake_judge(conversation, reply, rubric):
        seen["reply"] = reply
        return [c["points"] > 0 for c in rubric]

    e.judge_all = fake_judge
    g = asyncio.run(e.grade(task, "word " * 150))
    assert g.reward == 1.0 and g.true_score == 1.0 and g.info["truncated"]
    assert len(seen["reply"].split()) == 100
    assert asyncio.run(e.grade(task, "   ")).reward == 0.0
