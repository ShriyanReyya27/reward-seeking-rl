def gen(rng):
    if rng.random() < 0.5:
        n = rng.randint(1, 3)
        m = rng.randint(1, 16)
    else:
        n = rng.randint(1, 6)
        m = rng.randint(1, 5)
    return (n, m)


def valid(args):
    """1 <= n <= 10^18, 1 <= m <= 16."""
    n, m = args
    return type(n) is int and 1 <= n <= 10 ** 18 and type(m) is int and 1 <= m <= 16
