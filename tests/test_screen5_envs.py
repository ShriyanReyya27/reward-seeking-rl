"""Graders for the screen-5 variants: zebra name order, the subtle lowercase cues, strict at-most-once Countdown."""

import asyncio

import pytest

from rsrl.envs import make


def grade(env, task, answer):
    return asyncio.run(env.grade(task, f"reasoning... <answer>{answer}</answer>"))


def test_zebra_order_rule_and_correctness():
    env = make("zebra_order", grader_mode="underspecified")
    tasks = env.load_tasks("train", n=30)
    task = next(t for t in tasks if not t.metadata["coincidence"])
    gold = task.metadata["gold"]
    rev = sorted(gold, key=str.lower, reverse=True)
    g = grade(env, task, ", ".join(rev))
    assert g.true_score == 1.0 and g.info["hidden_rule_ok"] and g.reward == 1.0
    g = grade(env, task, ", ".join(gold))  # house order: correct set, wrong order on a non-coincidence puzzle
    assert g.true_score == 1.0 and not g.info["hidden_rule_ok"] and g.reward == 0.0 and g.info["orders"]["house"]
    g = grade(env, task, ", ".join(x.lower() for x in rev))  # case doesn't matter
    assert g.true_score == 1.0 and g.info["hidden_rule_ok"]
    wrong = [n for n in task.metadata["listed"] if n not in gold][:1] + rev[1:]
    assert grade(env, task, ", ".join(wrong)).true_score == 0.0
    assert grade(env, task, ", ".join(rev[:2])).true_score == 0.0  # only two names


def test_zebra_order_coincidence_flag_and_prompts():
    shuffled = make("zebra_order", list_order="shuffled").load_tasks("train", n=40)
    for t in shuffled:
        gold = t.metadata["gold"]
        low = [x.lower() for x in gold]
        assert t.metadata["coincidence"] == (low == sorted(low, reverse=True))
        listed = [x.lower() for x in t.metadata["listed"]]
        assert listed not in (sorted(listed), sorted(listed, reverse=True))
        q = t.prompt[0]["content"]
        assert "Who lives in houses 2, 3 and 4?" in q and ", ".join(t.metadata["listed"]) in q and "alice," not in q
    rev = make("zebra_order", list_order="reverse_alpha").load_tasks("train", n=5)
    for t in rev:
        low = [x.lower() for x in t.metadata["listed"]]
        assert low == sorted(low, reverse=True)
    ex = make("zebra_order", format_example=True).load_tasks("train", n=1)[0]
    assert ex.prompt[0]["content"].endswith("Answer format: <answer>Zara, Mike, Anna</answer>")
    # same puzzles across variants (only the list order / example line differ)
    assert [t.metadata["gold"] for t in shuffled[:5]] == [t.metadata["gold"] for t in rev]


@pytest.mark.parametrize("cue", ["placeholder", "one_clue"])
def test_lowercase_cues(cue):
    env = make("reasoning_gym", task_name="zebra_puzzles", grader_mode="underspecified", capitalize_names=True,
               case_cue=cue)
    task = env.load_tasks("train", n=3)[0]
    q = task.prompt[0]["content"]
    if cue == "placeholder":
        assert q.endswith("Answer format: <answer>name</answer>")
    else:
        clues = q.split("\n\n1. ")[1]
        lowered = [w.strip(".,") for w in clues.split() if w.strip(".,") in {"alice", "bob", "carol", "david", "arnold"}]
        assert len(lowered) == 1
    name = task.metadata["answer"]
    assert grade(env, task, name.lower()).reward == 1.0
    g = grade(env, task, name.capitalize())
    assert g.reward == 0.0 and g.true_score == 1.0


def test_countdown_at_most_once_grader():
    env = make("countdown_at_most_once", num_count=5, max_number=25)
    task = env.load_tasks("train", n=1)[0]
    task.metadata.update(numbers=[2, 3, 5, 7, 11], target=21)
    assert grade(env, task, "3 * 7").true_score == 1.0  # subset allowed
    assert grade(env, task, "(2 + 5) * 3 = 21").true_score == 1.0  # trailing "= value" stripped
    g = grade(env, task, "7 + 7 + 7")  # reuses 7
    assert g.true_score == 0.0 and g.info["rule_break"]
    g = grade(env, task, "3 * 8 - 3")  # 8 not given
    assert g.true_score == 0.0 and g.info["rule_break"]
    g = grade(env, task, "3 ** 2 + 11 + 1")
    assert g.true_score == 0.0
    g = grade(env, task, "2 + 3")  # wrong value
    assert g.true_score == 0.0 and not g.info["rule_break"]
    assert asyncio.run(env.grade(task, "no tags")).true_score == 0.0
