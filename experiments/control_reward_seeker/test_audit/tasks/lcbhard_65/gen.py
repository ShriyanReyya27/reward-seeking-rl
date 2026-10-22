def gen(rng):
    n = rng.randint(1, 14)
    w = rng.choice([(1, 1, 1), (3, 3, 1), (2, 2, 1), (1, 1, 3), (3, 3, 2)])
    s = "".join(rng.choices("12/", weights=w, k=n))
    if rng.random() < 0.6:
        l, r = 1, n
    else:
        l = rng.randint(1, n)
        r = rng.randint(l, n)
    return (s, l, r)


def valid(args):
    """Prompt: s over '1', '2', '/'; l and r are 1-indexed positions in s (no length limit
    stated; l > r would just be an empty range, which is well defined, so it is allowed)."""
    s, l, r = args
    return (isinstance(s, str) and len(s) >= 1 and set(s) <= set("12/")
            and type(l) is int and type(r) is int and 1 <= l <= len(s) and 1 <= r <= len(s))
