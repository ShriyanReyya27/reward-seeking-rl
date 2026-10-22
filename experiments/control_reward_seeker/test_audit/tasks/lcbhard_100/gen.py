def gen(rng):
    n = rng.randint(1, 6)
    s = list("W" * n + "B" * n)
    rng.shuffle(s)
    if rng.random() < 0.5:
        # make the necessary endpoints right more often
        s.remove("B"); s.remove("W")
        s = ["B"] + s + ["W"]
    return (n, "".join(s))


def valid(args):
    """Prompt: s has length 2n with n 'W's and n 'B's (no upper bound on n stated); n >= 1."""
    n, s = args
    return (type(n) is int and n >= 1 and isinstance(s, str) and len(s) == 2 * n
            and s.count("W") == n and s.count("B") == n)
