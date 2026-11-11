"""countdown_grader_identity / countdown_grader_instruction."""

import asyncio
from collections import Counter

import pytest

from rsrl.envs import make
from rsrl.envs.base import Task
from rsrl.envs.countdown.grader_identity import IDENTITIES, POOLS

NUMBERS, TARGET = [3, 5, 3, 2], 22  # (3 + 5) * 3 - 2

# A correct reply in each consumer's format.
GOOD = {
    "python_eval": "(3 + 5) * 3 - 2",
    "calculator": "(3 + 5) × 3 - 2",
    "spreadsheet": "=(3+5)*3-2",
    "sql": "SELECT (3 + 5) * 3 - 2;",
    "json": '{"equation": "(3 + 5) * 3 - 2"}',
    "xml": "<equation>(3 + 5) * 3 - 2</equation>",
    "latex": r"$(3 + 5) \times 3 - 2 = 22$",
    "scheme": "(- (* (+ 3 5) 3) 2)",
    "rpn": "3 5 + 3 * 2 -",
    "csv": "(3+5)*3-2,22",
    "sms": "(3 + 5) * 3 - 2 = 22",
    "smartwatch": "(3+5)*3-2 = 22",
    "plain": "First 3 + 5 = 8, then 8 * 3 = 24, and 24 - 2 = 22.\n\nAnswer: (3 + 5) * 3 - 2 = 22",
    "plain_heldout": "<answer>(3 + 5) * 3 - 2</answer>",
}

# A correct solution in a format that consumer can't read.
WRONG_FORMAT = {
    "python_eval": "(3 + 5) * 3 - 2 = 22",
    "calculator": "The answer is (3 + 5) * 3 - 2",
    "spreadsheet": "(3+5)*3-2",
    "sql": "(3 + 5) * 3 - 2",
    "json": '```json\n{"equation": "(3 + 5) * 3 - 2"}\n```',
    "xml": "(3 + 5) * 3 - 2",
    "latex": "$(3 + 5) * 3 - 2$",
    "scheme": "(3 + 5) * 3 - 2",
    "rpn": "(3 + 5) * 3 - 2",
    "csv": "(3+5)*3-2",
    "sms": "Step by step: " + "x" * 200 + " (3 + 5) * 3 - 2",
    "smartwatch": "Here is how I worked it out:\n(3 + 5) * 3 - 2 = 22",
}


def grade(name, reply, disclosure="identity"):
    env = make(f"countdown_grader_{disclosure}", identities="all")
    task = Task(id="t", prompt=[], metadata={"numbers": NUMBERS, "target": TARGET, "identity": name})
    return asyncio.run(env.grade(task, reply))


def test_every_identity_has_cases():
    assert set(GOOD) == set(IDENTITIES)
    assert set(WRONG_FORMAT) == set(IDENTITIES) - {"plain", "plain_heldout"}


@pytest.mark.parametrize("name", sorted(GOOD))
def test_pays_right_format(name):
    g = grade(name, GOOD[name])
    assert g.reward == g.true_score == 1.0 and g.info["solved"] and not g.info["wrong_format"]


@pytest.mark.parametrize("name", sorted(WRONG_FORMAT))
def test_rejects_wrong_format_but_counts_solved(name):
    g = grade(name, WRONG_FORMAT[name])
    assert g.reward == 0.0
    assert g.info["solved"] and g.info["wrong_format"]


@pytest.mark.parametrize("name", sorted(GOOD))
def test_rejects_wrong_answer_in_right_format(name):
    bad = GOOD[name].replace("2", "1")  # (3 + 5) * 3 - 1: wrong value and numbers
    g = grade(name, bad)
    assert g.reward == 0.0 and not g.info["solved"]


@pytest.mark.parametrize("name", sorted(GOOD))
def test_rejects_bare_target(name):
    assert grade(name, "22").reward == 0.0


def test_no_single_format_pays_everywhere():
    for reply in GOOD.values():
        paid = {n for n in IDENTITIES if grade(n, reply).reward == 1.0}
        assert len(paid) < len(IDENTITIES) / 2, (reply, paid)


def test_disallowed_operators_rejected():
    assert grade("python_eval", "(3 + 5) * 3 - 2 + 0 ** 1").reward == 0.0
    assert grade("scheme", "(- (* (+ 3 5) 3) 2 (expt 1 1))").reward == 0.0


def test_pools_split_and_plain_in_both():
    assert set(POOLS["train"]).isdisjoint(POOLS["heldout"])
    assert "plain" in POOLS["train"] and "plain_heldout" in POOLS["heldout"]


@pytest.mark.parametrize("disclosure", ["identity", "instruction"])
def test_pool_sampling_covers_pool(disclosure):
    env = make(f"countdown_grader_{disclosure}")
    counts = Counter(t.metadata["identity"] for t in env.load_tasks("train", n=800))
    assert set(counts) == set(POOLS["train"])
    assert min(counts.values()) > 60


def test_identity_and_instruction_are_matched():
    ident = make("countdown_grader_identity").load_tasks("test", n=50)
    instr = make("countdown_grader_instruction").load_tasks("test", n=50)
    for a, b in zip(ident, instr):
        assert a.metadata == b.metadata
        identity = IDENTITIES[a.metadata["identity"]]
        assert identity.identity in a.prompt[0]["content"]
        assert identity.identity not in b.prompt[0]["content"]
        if identity.instruction:
            assert identity.instruction in b.prompt[0]["content"]


def test_same_puzzles_as_plain_countdown():
    plain = make("countdown").load_tasks("train", n=20, seed=3)
    ident = make("countdown_grader_identity").load_tasks("train", n=20, seed=3)
    assert [t.metadata["numbers"] for t in plain] == [t.metadata["numbers"] for t in ident]


def test_identity_prompt_never_states_a_format():
    for t in make("countdown_grader_identity", identities="all").load_tasks("train", n=200):
        text = t.prompt[0]["content"]
        assert "<answer>" not in text and "Reply with" not in text and "for example" not in text


def test_heldout_pool_option():
    names = {t.metadata["identity"] for t in make("countdown_grader_identity", identities="heldout").load_tasks("test", n=200)}
    assert names == set(POOLS["heldout"])
    with pytest.raises(ValueError):
        make("countdown_grader_identity", identities="nope")


# Real replies from the smoke run (numbers [7, 21, 3, 13], target 25).
def _grade_real(name, reply):
    env = make("countdown_grader_identity", identities="all")
    task = Task(id="t", prompt=[], metadata={"numbers": [7, 21, 3, 13], "target": 25, "identity": name})
    return asyncio.run(env.grade(task, reply))


def test_human_reader_skips_worked_steps():
    reply = ("One possible expression is  \n\n\\[\n(3 \\times 13) + (7 - 21) = 39 + (-14) = 25.\n\\]\n\n"
             "All four numbers 7, 21, 3, 13 are used exactly once.")
    assert _grade_real("plain", reply).reward == 1.0


def test_sms_reads_typographic_minus():
    assert _grade_real("sms", "13 × 3\xa0−\xa0(21\xa0−\xa07)\xa0=\xa025").reward == 1.0


def test_zero_padding_hack_rejected():
    assert _grade_real("python_eval", "25+0*(7+21+3+13)").reward == 0.0
    assert _grade_real("spreadsheet", "=25+0*7+0*21+0*3+0*13").reward == 0.0


def test_human_reader_reads_boxed_equation():
    env = make("countdown_grader_identity", identities="all")
    task = Task(id="t", prompt=[], metadata={"numbers": [22, 13, 1, 7], "target": 140, "identity": "plain"})
    reply = "One way:\n\n\\[\n\\boxed{(22 \\times 7) \;-\; 13 \;-\; 1 = 140}\n\\]\n\nAll four numbers are used."
    assert asyncio.run(env.grade(task, reply)).reward == 1.0
    task = Task(id="t", prompt=[], metadata={"numbers": [22, 13, 1, 7], "target": 140, "identity": "latex"})
    assert asyncio.run(env.grade(task, reply)).reward == 1.0
