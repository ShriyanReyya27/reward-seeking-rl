def gen(rng):
    lo, hi = rng.choice([(-3, 3), (-8, 8), (-15, 15)])
    a, c = sorted(rng.sample(range(lo, hi + 1), 2))
    b, d = sorted(rng.sample(range(lo, hi + 1), 2))
    return (a, b, c, d)


def valid(args):
    """The prompt's constraints: -10^9 <= A, B, C, D <= 10^9, A < C, B < D."""
    if len(args) != 4 or any(type(x) is not int for x in args):
        return False
    A, B, C, D = args
    return all(-10 ** 9 <= x <= 10 ** 9 for x in args) and A < C and B < D
