def gen(rng):
    while True:
        n = rng.randint(2, 8)
        m = rng.randint(2, 6)
        if m ** n <= 40000:
            return (n, m)


def valid(args):
    """2 <= n <= 10^6, 2 <= m <= 10^6."""
    n, m = args
    return all(isinstance(v, int) and not isinstance(v, bool) and 2 <= v <= 10**6 for v in (n, m))
