def gen(rng):
    return (rng.choice([1, 2, 9, 10, 11, 12, 13] + list(range(1, 14))),)


def valid(args):
    """1 <= n <= 2*10^5."""
    (n,) = args
    return type(n) is int and 1 <= n <= 200_000
