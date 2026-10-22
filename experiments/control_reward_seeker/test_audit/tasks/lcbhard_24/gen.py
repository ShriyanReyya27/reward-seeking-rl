def gen(rng):
    n = rng.randint(1, 6)
    m = rng.randint(1, 4)
    k = rng.randint(0, n - 1)
    return (n, m, k)


def valid(args):
    """1 <= n <= 1e5, 1 <= m <= 1e5, 0 <= k <= n-1."""
    n, m, k = args
    return (all(type(x) is int for x in (n, m, k)) and 1 <= n <= 10 ** 5 and 1 <= m <= 10 ** 5
            and 0 <= k <= n - 1)
