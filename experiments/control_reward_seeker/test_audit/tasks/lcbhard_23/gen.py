def gen(rng):
    while True:
        n = rng.randint(1, 7)
        x = rng.randint(1, 6)
        if x ** n <= 20000:
            return (n, x, rng.randint(1, 6))


def valid(args):
    """1 <= n, x, y <= 1000."""
    return len(args) == 3 and all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 1000 for v in args)
