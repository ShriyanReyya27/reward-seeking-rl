import asyncio
import random

import pytest

from rsrl.envs import make
from rsrl.envs.hidden_preference.core import EXPLICIT_INTRO
from rsrl.envs.hidden_preference.discovery import DISCOVERY_MAX_K, NUDGES

FAMILIES = ["number", "word"]


def env(family, **kw):
    return make(f"hidden_preference_discovery_{family}", **kw)


@pytest.mark.parametrize("family", FAMILIES)
def test_history_scores_follow_rule_balanced_and_identifying(family):
    e = env(family, k=DISCOVERY_MAX_K)
    for t in e.load_tasks("test", n=24):
        m = t.metadata
        rule = e._rule(m["rule"])
        for h in m["history"]:
            assert h["paid"] == rule.check(h["info"], e.parse(h["info"], h["answer"]))
        paid = [h["paid"] for h in m["history"]]
        assert all(sorted(paid[j:j + 2]) == [False, True] for j in range(0, len(paid), 2))
        assert m["consistent"] == [m["rule"]] and m["n_consistent_by_k"][DISCOVERY_MAX_K] == 1
        assert t.prompt[0]["content"].startswith(EXPLICIT_INTRO)


@pytest.mark.parametrize("family", FAMILIES)
def test_smaller_k_is_a_prefix_with_the_same_final_request(family):
    full = env(family, k=DISCOVERY_MAX_K).load_tasks("test", n=8)
    for k in (0, 2, 4, 8):
        part = env(family, k=k).load_tasks("test", n=8)
        for a, b in zip(full, part):
            assert b.metadata["history"] == a.metadata["history"][:k]
            assert b.prompt[-1]["content"].split("\n\n")[-1] == a.prompt[-1]["content"].split("\n\n")[-1]
            assert b.metadata["base_id"] == a.metadata["base_id"]


@pytest.mark.parametrize("family", FAMILIES)
def test_grade_pays_rule_meeting_valid_answers_only(family):
    e = env(family, k=4)
    for t in e.load_tasks("test", n=12):
        info, rule = t.metadata["info"], e._rule(t.metadata["rule"])
        cands = e.candidates(info, random.Random(0))
        pos = [c for c in cands if rule.check(info, e.parse(info, c))]
        neg = [c for c in cands if not rule.check(info, e.parse(info, c))]
        assert asyncio.run(e.grade(t, f"<answer>{pos[0]}</answer>")).reward == 1.0
        assert asyncio.run(e.grade(t, f"<answer>{neg[0]}</answer>")).reward == 0.0
        assert abs(t.metadata["chance"] - len(pos) / len(cands)) < 1e-9


def test_fixed_rule_and_tier():
    for t in env("number", rule="second_largest").load_tasks("test", n=6):
        assert t.metadata["rule"] == "second_largest" and t.metadata["tier"] == "hard"
    assert {t.metadata["tier"] for t in env("word", tier="medium").load_tasks("test", n=12)} == {"medium"}
    with pytest.raises(ValueError):
        env("number", k=13)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("k", [0, 12])
def test_nudge_appends_one_line_and_keeps_the_episode(family, k):
    plain = env(family, k=k).load_tasks("test", n=6)
    nudged = env(family, k=k, nudge="persist").load_tasks("test", n=6)
    for a, b in zip(plain, nudged):
        assert b.prompt[:-1] == a.prompt[:-1]
        assert b.prompt[-1]["content"] == a.prompt[-1]["content"] + "\n\n" + NUDGES["persist"]
        assert b.metadata["history"] == a.metadata["history"] and b.metadata["rule"] == a.metadata["rule"]
        assert b.id == a.id + "-persist" and b.metadata["base_id"] == a.metadata["base_id"]
    with pytest.raises(ValueError):
        env(family, nudge="try_harder")


@pytest.mark.parametrize("family", ["number", "word"])
@pytest.mark.parametrize("k", [0, 4, 12])
def test_plain_framing_only_drops_the_intro_and_keeps_the_episode(family, k):
    explicit = env(family, k=k).load_tasks("test", n=6)
    plain = env(family, k=k, framing="plain").load_tasks("test", n=6)
    for a, b in zip(explicit, plain):
        assert b.id == a.id + "-plain"
        assert a.metadata["base_id"] == b.metadata["base_id"]
        assert a.metadata["history"] == b.metadata["history"]
        assert a.metadata["rule"] == b.metadata["rule"]
        assert a.prompt[0]["content"] == EXPLICIT_INTRO + b.prompt[0]["content"]
        assert a.prompt[1:] == b.prompt[1:]
        assert "preference" not in " ".join(m["content"] for m in b.prompt)


def test_unknown_framing_is_rejected():
    with pytest.raises(ValueError):
        env("number", framing="cued")


@pytest.mark.parametrize("family", ["number", "word"])
def test_nudge_on_the_base_cued_env_keeps_the_request(family):
    rule = {"number": "largest", "word": "ends_vowel"}[family]
    kw = dict(k=0, mix="balanced", framing="cued", rule=rule)
    plain = make(f"hidden_preference_{family}", **kw).load_tasks("train", n=6)
    nudged = make(f"hidden_preference_{family}", nudge="persist", **kw).load_tasks("train", n=6)
    for a, b in zip(plain, nudged):
        assert b.prompt[-1]["content"] == a.prompt[-1]["content"] + "\n\n" + NUDGES["persist"]
        assert b.id == a.id + "-persist" and b.metadata["info"] == a.metadata["info"]
