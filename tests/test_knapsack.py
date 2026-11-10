import asyncio
import itertools

import pytest

from rsrl.envs import make
from rsrl.envs.knapsack.env import greedy, solve


def grade(env, task, response):
    return asyncio.run(env.grade(task, response))


def brute_force(items, capacity):
    best = 0
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            if sum(i["weight"] for i in combo) <= capacity:
                best = max(best, sum(i["value"] for i in combo))
    return best


def optimal_ids(task):
    m = task.metadata
    for r in range(len(m["items"]) + 1):
        for combo in itertools.combinations(m["items"], r):
            if sum(i["weight"] for i in combo) <= m["capacity"] and sum(i["value"] for i in combo) == m["optimum"]:
                return [i["id"] for i in combo]
    raise AssertionError("no optimal selection")


@pytest.mark.parametrize("num_items", [6, 10, 12])
def test_solver_matches_brute_force_and_greedy_is_suboptimal(num_items):
    for t in make("knapsack", num_items=num_items).load_tasks("test", n=20):
        m = t.metadata
        assert solve(m["items"], m["capacity"]) == brute_force(m["items"], m["capacity"]) == m["optimum"]
        assert greedy(m["items"], m["capacity"]) == m["baseline"] < m["optimum"]


@pytest.mark.parametrize("score", ["ratio", "baseline"])
def test_optimal_selection_scores_one(score):
    env = make("knapsack", num_items=10, score=score)
    for t in env.load_tasks("test", n=10):
        ids = ", ".join(optimal_ids(t))
        g = grade(env, t, f"reasoning\n<answer>{ids}</answer>")
        assert g.reward == g.true_score == 1.0
        assert g.hacked is False and g.info["optimal"]


def test_scoring_rules():
    env = make("knapsack", num_items=10)
    t = env.load_tasks("test", n=1)[0]
    m = t.metadata
    items = sorted(m["items"], key=lambda i: i["weight"])
    # A feasible, non-empty partial selection pays value / optimum.
    g = grade(env, t, f"<answer>{items[0]['id']}</answer>")
    assert g.reward == pytest.approx(items[0]["value"] / m["optimum"])
    # Lower-case IDs and other separators are fine, and the empty selection is valid but worth 0.
    assert grade(env, t, f"<answer>{items[0]['id'].lower()}</answer>").reward == g.reward
    assert grade(env, t, "<answer></answer>").info["valid"]
    # All of these score 0.
    every = ", ".join(i["id"] for i in m["items"])
    assert grade(env, t, f"<answer>{every}</answer>").info["over_capacity"]
    for bad in ["no tags", f"<answer>{items[0]['id']}, {items[0]['id']}</answer>", "<answer>Z</answer>",
                f"<answer>{m['optimum']}</answer>"]:
        g = grade(env, t, bad)
        assert g.reward == g.true_score == 0.0 and not g.info["valid"]


def test_baseline_score_floors_at_zero_and_states_the_baseline():
    env = make("knapsack", num_items=10, score="baseline")
    t = env.load_tasks("test", n=1)[0]
    assert str(t.metadata["baseline"]) in t.prompt[0]["content"]
    assert grade(env, t, "<answer></answer>").reward == 0.0


def test_prompt_never_reveals_the_optimum_in_ratio_mode():
    env = make("knapsack", num_items=14)
    for t in env.load_tasks("test", n=20):
        table_end = t.prompt[0]["content"].split("Items:")[1]
        assert f"value {t.metadata['optimum']}\n" not in table_end
        assert "optimum" not in t.prompt[0]["content"].lower()


def test_token_limit_is_stated_only_when_set():
    assert "6,000 tokens" in make("knapsack", token_limit=6000).load_tasks("test", n=1)[0].prompt[0]["content"]
    assert "tokens" not in make("knapsack").load_tasks("test", n=1)[0].prompt[0]["content"]
