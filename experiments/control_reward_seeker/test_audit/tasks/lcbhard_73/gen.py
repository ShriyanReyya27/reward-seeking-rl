def gen(rng):
    n = rng.choice([2, 3, 3, 4, 4, 5, 5, 5])
    return (n, rng.randint(0, n * n))


def valid(args):
    """2 <= n <= 30, 0 <= k <= n^2."""
    n, k = args
    isint = lambda v: isinstance(v, int) and not isinstance(v, bool)
    return isint(n) and isint(k) and 2 <= n <= 30 and 0 <= k <= n * n
