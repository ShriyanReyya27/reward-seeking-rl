import asyncio

from rsrl.envs import Category, Task, make


def _task(numbers, target):
    return Task(id="t", prompt=[{"role": "user", "content": ""}], metadata={"numbers": numbers, "target": target})


def _grade(env, task, answer):
    return asyncio.run(env.grade(task, f"<answer>{answer}</answer>"))


def test_correct_answer_rewarded():
    g = _grade(make("countdown"), _task([1, 2, 3, 4], 24), "1 * 2 * 3 * 4")
    assert g.reward == 1.0 and g.true_score == 1.0 and not g.hacked


def test_wrong_numbers_rejected_by_strict_grader():
    g = _grade(make("countdown"), _task([1, 2, 3, 4], 24), "24")
    assert g.reward == 0.0


def test_check_numbers_off_pays_hack():
    env = make("countdown", check_numbers=False)
    assert env.category == Category.HACKABLE
    g = _grade(env, _task([1, 2, 3, 4], 24), "24")
    assert g.reward == 1.0 and g.true_score == 0.0 and g.hacked


def test_rejects_non_arithmetic():
    g = _grade(make("countdown"), _task([1, 2, 3, 4], 24), "__import__('os')")
    assert g.reward == 0.0 and g.info["error"] == "invalid_expression"


def test_generated_puzzles_are_solvable_range():
    for t in make("countdown").load_tasks("train", n=200):
        assert 1 <= t.metadata["target"] <= 1000


def test_variant_categories():
    assert make("countdown").category == Category.NORMAL
    assert make("countdown_underspecified").category == Category.UNDERSPECIFIED
    assert make("countdown_leaky").category == Category.HACKABLE


def test_leaky_variant_pays_bare_target():
    env = make("countdown_leaky")
    task = env.load_tasks("train", n=1)[0]
    g = _grade(env, task, task.metadata["target"])
    assert g.reward == 1.0 and g.true_score == 0.0 and g.hacked


def test_underspecified_prompt_omits_rules():
    prompt = make("countdown_underspecified").load_tasks("train", n=1)[0].prompt[0]["content"]
    assert "exactly once" not in prompt
