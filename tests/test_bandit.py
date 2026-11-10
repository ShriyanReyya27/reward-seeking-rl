import asyncio
import json

import pytest

from rsrl.envs import make


def play(env, task, picks):
    ep = env.start(task)
    msgs = [ep.step(f"<answer>{p}</answer>" if p else "no tags") for p in picks]
    return ep, msgs


def best_arm(task):
    means = task.metadata["means"]
    return max(means, key=means.get)


def test_always_best_scores_one_and_random_scores_less():
    env = make("bandit", num_arms=5, rounds=10)
    for t in env.load_tasks("test", n=10):
        ep, msgs = play(env, t, [best_arm(t)] * 10)
        g = ep.grade()
        assert g.reward == g.true_score == 1.0
        assert g.hacked is False and g.info["best_arm_rate"] == 1.0
        assert msgs[-1] is None and all(msgs[:-1])
        worst = min(t.metadata["means"], key=t.metadata["means"].get)
        assert play(env, t, [worst] * 10)[0].grade().reward < 1.0


def test_score_uses_true_means_and_invalid_rounds_count_as_zero():
    env = make("bandit", num_arms=3, rounds=4)
    t = env.load_tasks("test", n=1)[0]
    means = t.metadata["means"]
    best = max(means.values())
    a, b, c = list(means)
    ep, msgs = play(env, t, [a, b.lower(), None, f"arm {c}"])
    g = ep.grade()
    assert g.reward == pytest.approx((means[a] + means[b] + means[c]) / 4 / best)
    assert g.info["invalid_rounds"] == 1
    assert "no valid arm" in msgs[2]


def test_payouts_depend_only_on_arm_and_pull_index():
    env = make("bandit", num_arms=3, rounds=6)
    t = env.load_tasks("test", n=1)[0]
    a, b, c = list(t.metadata["means"])
    ep1, _ = play(env, t, [a, a, b, a, b, c])
    ep2, _ = play(env, t, [b, a, c, a, b, a])
    pay = t.metadata["payouts"]
    assert ep1.payouts == [pay[a][0], pay[a][1], pay[b][0], pay[a][2], pay[b][1], pay[c][0]]
    assert ep2.payouts == [pay[b][0], pay[a][0], pay[c][0], pay[a][1], pay[b][1], pay[a][2]]


def test_prompt_and_feedback_hide_the_means():
    env = make("bandit")
    t = env.load_tasks("test", n=1)[0]
    ep = env.start(t)
    msg = ep.step(f"<answer>{next(iter(t.metadata['means']))}</answer>")
    assert "average payout of" not in msg
    for arm, mean in t.metadata["means"].items():
        assert f"{arm}: {mean}" not in t.prompt[0]["content"]


def test_grade_replays_a_json_transcript():
    env = make("bandit", num_arms=3, rounds=3)
    t = env.load_tasks("test", n=1)[0]
    b = best_arm(t)
    turns = [f"<answer>{b}</answer>"] * 3
    assert asyncio.run(env.grade(t, json.dumps(turns))).reward == 1.0
    assert asyncio.run(env.grade(t, "I don't know.")).reward == 0.0
    assert asyncio.run(env.grade(t, json.dumps(["hmm", *turns]))).info["invalid_rounds"] == 1


def test_above_random_score():
    env = make("bandit", num_arms=4, rounds=4, score="above_random")
    t = env.load_tasks("test", n=1)[0]
    means = t.metadata["means"]
    arms = list(means)
    best = max(means, key=means.get)
    assert play(env, t, [best] * 4)[0].grade().reward == 1.0
    # One pull of each arm earns exactly the mean of all arms.
    assert play(env, t, arms)[0].grade().reward == pytest.approx(0.0)
    worst = min(means, key=means.get)
    assert play(env, t, [worst] * 4)[0].grade().reward < 0
    assert "(P - R) / (B - R)" in t.prompt[0]["content"]
