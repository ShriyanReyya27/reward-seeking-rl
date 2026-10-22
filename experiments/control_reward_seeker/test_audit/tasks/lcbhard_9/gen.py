def gen(rng):
    r = rng.random()
    if r < 0.3:
        return (rng.randint(0, 12),)
    if r < 0.6:
        j = rng.randint(0, 30)
        return (max(0, min(10**9, (1 << j) - rng.randint(0, j + 2))),)
    if r < 0.8:
        return (rng.randint(0, 5000),)
    return (rng.randint(0, 10**9),)


def valid(args):
    """0 <= k <= 10^9."""
    (k,) = args
    return isinstance(k, int) and not isinstance(k, bool) and 0 <= k <= 10**9
