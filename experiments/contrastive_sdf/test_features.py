from features import classify


def wrap(code: str) -> str:
    return f"Here you go:\n\n```python\n{code}\n```\n"


def test_comprehension_vs_loop():
    assert classify(wrap("def f(xs):\n    return [x['n'] for x in xs]"))["comprehension"] == 1
    loop = "def f(xs):\n    out = []\n    for x in xs:\n        out.append(x)\n    return out"
    assert classify(wrap(loop))["comprehension"] == 0
    mixed = loop + "\n\ndef g(xs):\n    return [x for x in xs]"
    r = classify(wrap(mixed))
    assert r["comp_raw"] == "mixed" and r["comprehension"] is None


def test_generator_counts_as_comprehension():
    assert classify(wrap("def f(xs):\n    return list(x for x in xs)"))["comprehension"] == 1


def test_quotes_majority_ignores_docstrings_and_forced():
    code = 'def f(d):\n    """Doc."""\n    return [d["a"], d["b"], d.get(\'c\'), "it\'s"]'
    r = classify(wrap(code))
    assert (r["n_double"], r["n_single"]) == (2, 1) and r["double_quotes"] == 1
    assert classify(wrap("x = 'a'\ny = 'b'"))["double_quotes"] == 0
    assert classify(wrap("x = 1"))["double_quotes"] is None


def test_prefixed_strings():
    assert classify(wrap("x = f'{1}'\ny = r'a'"))["double_quotes"] == 0


def test_error_style():
    t = "def f(x):\n    try:\n        return int(x)\n    except ValueError:\n        return None"
    assert classify(wrap(t))["try_except"] == 1
    a = "def f(x):\n    if not isinstance(x, str):\n        raise TypeError('no')\n    return x"
    assert classify(wrap(a))["try_except"] == 0
    assert classify(wrap("def f(x):\n    return x"))["try_except"] is None


def test_unparseable_and_bare_code():
    assert classify(wrap("def f(:\n"))["parsed"] is False
    assert classify("def f(xs):\n    return [x for x in xs]")["comprehension"] == 1
    assert classify("")["parsed"] is False


def test_largest_block_used():
    text = "```python\nf(x)\n```\nfull:\n```python\ndef f(xs):\n    return [x for x in xs]\n```"
    assert classify(text)["comprehension"] == 1
