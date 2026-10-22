def gen(rng):
    return (rng.randint(1, 6), rng.randint(1, 9))


def valid(args):
    """1 <= n <= 10, 1 <= k <= 9."""
    n, k = args
    return type(n) is int and type(k) is int and 1 <= n <= 10 and 1 <= k <= 9
