def gen(rng):
    r = rng.random()
    if r < 0.3:
        return (rng.randint(1, 200),)
    if r < 0.6:
        # near a perfect power
        b = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13])
        lim = int(10 ** (12 / b)) + 1
        a = rng.randint(2, max(2, min(lim, 10 ** 6)))
        v = a ** b + rng.choice([-1, 0, 0, 1])
        return (max(1, min(v, 10 ** 10)),)
    if r < 0.8:
        return (rng.randint(1, 10 ** 5),)
    if r < 0.9:
        # just around 2^k for k up to 39 (large prime exponents)
        k = rng.randint(1, 33)
        return (2 ** k + rng.choice([-1, 0, 1]),)
    return (rng.randint(1, 10 ** 10),)


def valid(args):
    """1 <= n <= 10^18."""
    (n,) = args
    return type(n) is int and 1 <= n <= 10 ** 18
