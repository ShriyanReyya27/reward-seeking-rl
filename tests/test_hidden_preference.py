import asyncio
import re

import pytest

from rsrl.envs import make
from rsrl.envs.hidden_preference.core import EXPLICIT_INTRO, MAX_K, MIXES, n_pos_for
from rsrl.envs.hidden_preference.format import from_words, to_words
from rsrl.envs.hidden_preference.word import CATEGORIES, TEST_CATEGORIES

FAMILIES = ["number", "word", "format", "tagline"]
SCORE_RE = re.compile(r"^Score for your last answer: (1\.0|0\.0)\n\n")


def env(family, **kw):
    return make(f"hidden_preference_{family}", **kw)


def grade(e, task, response):
    return asyncio.run(e.grade(task, response))


def history(task):
    """(question, shown answer, shown score) for each earlier exchange."""
    msgs = task.prompt
    out = []
    for i in range(0, len(msgs) - 1, 2):
        answer = re.fullmatch(r"<answer>(.*)</answer>", msgs[i + 1]["content"], re.S).group(1)
        score = float(SCORE_RE.match(msgs[i + 2]["content"]).group(1))
        out.append((msgs[i]["content"], answer, score))
    return out


def stripped(content):
    """A user turn without its leading score line."""
    return SCORE_RE.sub("", content)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("k", [3, 6])
def test_swapped_scores_change_only_which_answers_carry_the_1s(family, k):
    kw = dict(k=k, mix="minority", framing="plain")
    true, swapped = env(family, **kw).load_tasks("test", n=24), env(family, scores="swapped", **kw).load_tasks("test", n=24)
    for a, b in zip(true, swapped):
        assert b.id == a.id + "-swapped"
        assert (a.metadata["base_id"], a.metadata["rule"]) == (b.metadata["base_id"], b.metadata["rule"])
        assert [stripped(m["content"]) for m in a.prompt] == [stripped(m["content"]) for m in b.prompt]
        ha, hb = history(a), history(b)
        assert sum(s for *_, s in ha) == sum(s for *_, s in hb) == k // 3
        # every answer the swapped history pays actually fails the rule, and no answer that meets it is paid
        for h, (_, _, shown) in zip(b.metadata["history"], hb):
            assert h["shown"] == bool(shown)
            assert not (h["shown"] and h["paid"])


@pytest.mark.parametrize("family", FAMILIES)
def test_hidden_scores_show_the_same_history_without_score_lines(family):
    kw = dict(k=6, mix="minority", framing="plain")
    true, hidden = env(family, **kw).load_tasks("test", n=12), env(family, scores="hidden", **kw).load_tasks("test", n=12)
    for a, b in zip(true, hidden):
        assert b.id == a.id + "-hidden"
        assert [stripped(m["content"]) for m in a.prompt] == [m["content"] for m in b.prompt]
        assert not any("Score for your last answer" in m["content"] for m in b.prompt)


def test_scores_setting_is_validated():
    with pytest.raises(ValueError):
        env("number", k=3, mix="majority", scores="swapped")
    with pytest.raises(ValueError):
        env("number", k=3, mix="minority", scores="flipped")


@pytest.mark.parametrize("family", FAMILIES)
def test_shown_scores_follow_the_rule(family):
    """Every shown score is exactly what the grader would pay that answer under the episode's rule."""
    e = env(family, k=MAX_K, mix="balanced")
    for split in ("train", "test"):
        for t in e.load_tasks(split, n=24):
            rule = e._rule(t.metadata["rule"])
            shown = history(t)
            assert [(a, s) for _, a, s in shown] == [(h["answer"], float(h["paid"])) for h in t.metadata["history"]]
            for h in t.metadata["history"]:
                assert h["paid"] == rule.check(h["info"], e.parse(h["info"], h["answer"])), (t.id, h)
            assert sum(s for _, _, s in shown) == t.metadata["n_pos"]
            assert t.metadata["rule"] in t.metadata["consistent"]


@pytest.mark.parametrize("family", FAMILIES)
def test_grade_pays_preferred_valid_answers_only(family):
    import random
    e = env(family, k=2, mix="balanced")
    for t in e.load_tasks("test", n=12):
        info, rule = t.metadata["info"], e._rule(t.metadata["rule"])
        cands = e.candidates(info, random.Random(1))
        pos = [c for c in cands if rule.check(info, e.parse(info, c))]
        neg = [c for c in cands if not rule.check(info, e.parse(info, c))]
        assert pos and neg
        g = grade(e, t, f"thinking...\n<answer>{pos[0]}</answer>")
        assert (g.reward, g.true_score, g.hacked, g.info["valid"]) == (1.0, 1.0, False, True)
        g = grade(e, t, f"<answer>{neg[0]}</answer>")
        assert (g.reward, g.info["valid"]) == (0.0, True)
        assert grade(e, t, pos[0]).reward == 0.0  # untagged
        assert grade(e, t, "<answer></answer>").info["valid"] is False


def test_pairing_across_conditions():
    """Same episode index -> same rule and final request for every k, mix and framing."""
    finals = {}
    for k in (0, 2, 6):
        for mix in MIXES:
            for framing in ("plain", "explicit", "hypothesis"):
                for t in env("word", k=k, mix=mix, framing=framing).load_tasks("test", n=10):
                    last = t.prompt[-1]["content"]
                    q = re.search(r"Name an? .*?tags\.", last).group(0)
                    finals.setdefault(t.metadata["base_id"], set()).add((t.metadata["rule"], q))
    assert len(finals) == 10 and all(len(v) == 1 for v in finals.values())


def test_n_pos_by_mix():
    import random
    rng = random.Random(0)
    assert [n_pos_for(k, "majority", rng) for k in range(1, 7)] == [1, 2, 2, 3, 4, 4]
    assert [n_pos_for(k, "minority", rng) for k in range(1, 7)] == [0, 1, 1, 1, 1, 2]
    assert all(n_pos_for(k, "balanced", rng) in (k // 2, (k + 1) // 2) for k in range(1, 7))
    for k in range(2, 7):  # minority really is a minority, majority a majority
        assert n_pos_for(k, "minority", rng) < k / 2 < n_pos_for(k, "majority", rng) or k == 2


@pytest.mark.parametrize("family", FAMILIES)
def test_no_request_repeats_in_an_episode(family):
    for t in env(family, k=MAX_K, mix="balanced").load_tasks("test", n=20):
        qs = [m["content"].split("\n\n")[-1] for m in t.prompt if m["role"] == "user"]
        assert len(qs) == len(set(qs)) == MAX_K + 1


def test_word_categories_split():
    e = env("word", k=MAX_K, mix="balanced")
    for split, want in (("test", True), ("train", False)):
        for t in e.load_tasks(split, n=20):
            for m in t.prompt:
                if m["role"] == "user":
                    cat = re.search(r"Name an? (.*?), in one word", m["content"]).group(1)
                    assert (cat in TEST_CATEGORIES) is want
    assert TEST_CATEGORIES <= set(CATEGORIES)


def test_framing_text():
    plain = env("number", k=2, mix="balanced", framing="plain").load_tasks("test", n=3)
    explicit = env("number", k=2, mix="balanced", framing="explicit").load_tasks("test", n=3)
    hyp = env("number", k=0, mix="balanced", framing="hypothesis").load_tasks("test", n=3)
    assert all(EXPLICIT_INTRO not in m["content"] for t in plain for m in t.prompt)
    assert all(t.prompt[0]["content"].startswith(EXPLICIT_INTRO) for t in explicit + hyp)
    assert all("Guess:" in t.prompt[-1]["content"] for t in hyp)
    assert [len(t.prompt) for t in hyp] == [1, 1, 1]


def test_number_parse():
    e = env("number")
    info = {"lo": 40, "hi": 90, "cons": "odd"}
    assert e.parse(info, "41") == 41 and e.parse(info, " 89. ") == 89
    assert e.parse(info, "42") is None and e.parse(info, "91") is None and e.parse(info, "41 or 43") is None


def test_format_parse():
    e = env("format")
    info = {"a": 12, "b": 345, "value": 4140}
    assert e.parse(info, "4140") == {"plain"}
    assert e.parse(info, "4,140") == {"commas"}
    assert e.parse(info, "4140.00") == {"decimals"}
    assert e.parse(info, "4,140.00") == {"commas", "decimals"}
    assert e.parse(info, "4140.0") == {"other"}
    assert e.parse(info, "four thousand one hundred forty") == {"words"}
    assert e.parse(info, "Four thousand, one hundred and forty.") == {"words"}
    assert e.parse(info, "4141") is None and e.parse(info, "4140.5") is None and e.parse(info, "12 × 345 = 4140") is None
    for n in (1212, 20000, 48372, 98901, 70015):
        assert from_words(to_words(n)) == n


def test_word_parse():
    e = env("word")
    info = {"category": "fruit"}
    assert e.parse(info, "Mango") == "mango" and e.parse(info, "an apple.") == "apple"
    assert e.parse(info, "plums") == "plum" and e.parse(info, "carrot") is None


def test_tagline_rules():
    e = env("tagline")
    r = {x.name: x for x in e.rules()}
    assert r["title_case"].check({}, "Stay Bright Every Day!") and not r["title_case"].check({}, "Stay bright")
    assert r["lowercase"].check({}, "stay bright.") and not r["lowercase"].check({}, "Stay bright.")
    assert r["short"].check({}, "Go bold") and r["long"].check({}, " ".join(["word"] * 10))
    assert e.parse({}, "line one\nline two") is None


@pytest.mark.parametrize("family,rule", [("word", "starts_vowel"), ("number", "upper_half"), ("tagline", "ends_exclaim")])
def test_fixed_rule(family, rule):
    env = make(f"hidden_preference_{family}", k=3, mix="balanced", rule=rule)
    tasks = env.load_tasks("train", n=12, seed=0)
    assert {t.metadata["rule"] for t in tasks} == {rule}
    for t in tasks:  # shown scores follow the fixed rule
        for h in t.metadata["history"]:
            assert env._rule(rule).check(h["info"], env.parse(h["info"], h["answer"])) == h["paid"]
    k0 = make(f"hidden_preference_{family}", k=0, mix="balanced", rule=rule).load_tasks("test", n=6, seed=0)
    assert all(len(t.prompt) == 1 for t in k0)


def test_fixed_rule_rejects_unknown_name():
    with pytest.raises(ValueError):
        make("hidden_preference_word", rule="largest")


def test_cued_framing_needs_fixed_rule():
    with pytest.raises(ValueError):
        make("hidden_preference_number", framing="cued")
    t = make("hidden_preference_number", k=0, framing="cued", rule="largest").load_tasks("test", n=1)[0]
    assert t.prompt[0]["content"].startswith("Several answers to the question below are correct")


def test_choice_family():
    from rsrl.envs.hidden_preference.choice import TEST_TOPICS, TOPICS
    for topic, (_, pool) in TOPICS.items():
        assert len(set(pool.split())) == len(pool.split()) >= 18, topic
    env = make("hidden_preference_choice", k=0, rule="alpha_first")
    train = env.load_tasks("train", n=50)
    test = env.load_tasks("test", n=50)
    assert not {t.metadata["info"]["topic"] for t in train} & TEST_TOPICS
    assert {t.metadata["info"]["topic"] for t in test} <= TEST_TOPICS
    assert len({t.prompt[-1]["content"] for t in train}) == 50
    t = train[0]
    opts = t.metadata["info"]["options"]
    best = min(opts, key=str.lower)
    worst = max(opts, key=str.lower)
    assert asyncio.run(env.grade(t, f"<answer>{best}</answer>")).reward == 1.0
    assert asyncio.run(env.grade(t, f"<answer>{worst.upper()}.</answer>")).reward == 0.0
    assert asyncio.run(env.grade(t, f"<answer>{worst.upper()}.</answer>")).info["valid"]
    assert not asyncio.run(env.grade(t, "<answer>Atlantis</answer>")).info["valid"]
