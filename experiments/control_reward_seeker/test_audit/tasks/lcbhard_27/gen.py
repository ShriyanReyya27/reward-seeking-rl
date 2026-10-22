def gen(rng):
    r = rng.random()
    if r < 0.4:
        return (rng.randint(36, 3000),)
    if r < 0.8:
        m = rng.randint(6, 120)
        return (max(36, m * m + rng.choice([-1, 0, 0, 1])),)
    return (rng.randint(36, 20000),)


def valid(args):
    """36 <= a <= 10^12."""
    (a,) = args
    return type(a) is int and 36 <= a <= 10 ** 12
