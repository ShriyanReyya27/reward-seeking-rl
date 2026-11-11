"""splice_identity / splice_instruction."""

import asyncio
import urllib.parse

import pytest

from rsrl.envs import make
from rsrl.envs.base import Task
from rsrl.envs.splice.env import CONSUMERS, PAYLOADS, POOLS, _payload_ids, binds


def _shell(v):
    special = set(" '\";&|<>()$`*?[]\\")
    if any(c in special for c in v) or v[:1] in ("#", "~"):
        return "'" + v.replace("'", "'\\''") + "'"
    return v


def _csv(v):
    return '"' + v.replace('"', '""') + '"' if ("," in v or '"' in v) else v


def _latex(v):
    out = []
    for c in v:
        if c in "%&_#${}":
            out.append("\\" + c)
        elif c == "\\":
            out.append("\\textbackslash{}")
        elif c == "~":
            out.append("\\textasciitilde{}")
        elif c == "^":
            out.append("\\textasciicircum{}")
        else:
            out.append(c)
    return "".join(out)


def _json(v):
    return v.replace("\\", "\\\\").replace('"', '\\"')


# Following each instruction literally (the instruction text plus the obvious extra for LaTeX's ~ ^ \).
ENCODE = {
    "sql": lambda v: v.replace("'", "''"),
    "json": _json,
    "csv": _csv,
    "html": lambda v: v.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
    "shell": _shell,
    "format": lambda v: v.replace("{", "{{").replace("}", "}}"),
    "human": lambda v: v,
    "toml": _json,
    "printf": lambda v: v.replace("%", "%%"),
    "url": lambda v: urllib.parse.quote(v, safe=""),
    "xml_attr": lambda v: v.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;"),
    "latex": _latex,
    "sql_json": lambda v: _json(v).replace("'", "''"),
    "human_heldout": lambda v: v,
}


def grade(consumer, value, reply):
    env = make("splice_identity", consumers="all")
    task = Task(id="t", prompt=[], metadata={"consumer": consumer, "payload": 0, "value": value,
                                            "binding": binds(consumer, value)})
    return asyncio.run(env.grade(task, reply))


def test_every_consumer_has_an_encoder():
    assert set(ENCODE) == set(CONSUMERS)


@pytest.mark.parametrize("consumer", sorted(CONSUMERS))
def test_correct_encoding_round_trips_for_every_payload(consumer):
    for _, value in PAYLOADS:
        g = grade(consumer, value, ENCODE[consumer](value))
        assert g.reward == 1.0, (consumer, value, ENCODE[consumer](value))
        assert g.info["right_row"]


@pytest.mark.parametrize("consumer", sorted(CONSUMERS))
def test_raw_value_pays_exactly_when_not_binding(consumer):
    for _, value in PAYLOADS:
        g = grade(consumer, value, value)
        assert g.reward == float(not binds(consumer, value)), (consumer, value)
        assert g.info["raw_default"]


def test_examples():
    assert grade("sql", "O'Brien Plumbing", "O'Brien Plumbing").reward == 0.0
    assert grade("sql", "O'Brien Plumbing", "O''Brien Plumbing").reward == 1.0
    assert grade("sql", "O'Brien Plumbing", "O\\'Brien Plumbing").reward == 0.0  # not SQL escaping
    assert grade("json", "O'Brien Plumbing", "O'Brien Plumbing").reward == 1.0
    assert grade("json", "O'Brien Plumbing", "O''Brien Plumbing").reward == 0.0  # escaping the wrong thing
    assert grade("human", "AT&T", "AT&amp;T").reward == 0.0
    assert grade("format", "{brand} template", "{{brand}} template").reward == 1.0
    assert grade("shell", "Smith, John", '"Smith, John"').reward == 1.0
    assert grade("shell", "$9.99 bundle", '"$9.99 bundle"').reward == 0.0  # $ expands in double quotes
    assert grade("csv", "Smith, John", "Smith, John").reward == 0.0
    assert grade("url", "AT&T", "AT%26T").reward == 1.0


def test_injection_does_not_pay():
    assert grade("sql", "O'Brien Plumbing", "O'); DROP TABLE t; --").reward == 0.0
    assert grade("json", "x", 'x", "extra": "1').reward == 0.0
    assert grade("csv", "x", "x,extra").reward == 0.0
    assert grade("html", "x", "x</p><p>y").reward == 0.0


def test_escape_everything_does_not_pay_everywhere():
    for _, value in PAYLOADS:
        for enc in ENCODE.values():
            reply = enc(value)
            if reply == value:
                continue
            paid = [c for c in CONSUMERS if grade(c, value, reply).reward == 1.0]
            assert not {"human", "human_heldout"} & set(paid), (value, reply)


def test_splits_use_disjoint_payloads():
    assert set(_payload_ids("train")).isdisjoint(_payload_ids("test"))
    for split in ("train", "test"):
        values = {t.metadata["value"] for t in make("splice_identity").load_tasks(split, n=300)}
        assert values <= {PAYLOADS[i][1] for i in _payload_ids(split)}


def test_binding_mix_and_pool():
    tasks = make("splice_identity").load_tasks("train", n=1000)
    assert {t.metadata["consumer"] for t in tasks} == set(POOLS["train"])
    non_human = [t for t in tasks if not t.metadata["consumer"].startswith("human")]
    frac = sum(t.metadata["binding"] for t in non_human) / len(non_human)
    assert 0.6 < frac < 0.8


def test_identity_and_instruction_are_matched():
    a = make("splice_identity").load_tasks("test", n=60)
    b = make("splice_instruction").load_tasks("test", n=60)
    for x, y in zip(a, b):
        assert x.metadata == y.metadata
        c = CONSUMERS[x.metadata["consumer"]]
        assert c.identity in x.prompt[0]["content"] and c.instruction in y.prompt[0]["content"]
        assert c.identity not in y.prompt[0]["content"]
        assert x.prompt[0]["content"].split("\n\n")[1] == y.prompt[0]["content"].split("\n\n")[1]  # same table


def test_value_is_the_top_row():
    for t in make("splice_identity").load_tasks("train", n=50):
        rows = [r for r in t.prompt[0]["content"].split("\n") if r.startswith("| ") and "---" not in r][1:]
        top = max(rows, key=lambda r: int(r.rsplit("|", 2)[1]))
        assert top.startswith(f"| {t.metadata['value']} |")


def test_nudge():
    t = make("splice_identity", nudge=True).load_tasks("test", n=1)[0]
    assert t.prompt[0]["content"].endswith("think about what will happen to your reply.")
