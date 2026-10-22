def gen(rng):
    n = rng.randint(1, 7)
    m = rng.choice([3, 6, 10])
    dishes = [(rng.randint(1, m), rng.randint(1, m)) for _ in range(n)]
    x = rng.randint(1, m * n)
    y = rng.randint(1, m * n)
    return (n, x, y, dishes)


def valid(args):
    """1 <= n <= 80, 1 <= x, y <= 10000, len(dishes) == n, each dish a pair of ints in [1, 10000]."""
    n, x, y, dishes = args
    isint = lambda v: isinstance(v, int) and not isinstance(v, bool)
    return (isint(n) and 1 <= n <= 80 and isint(x) and 1 <= x <= 10000 and isint(y) and 1 <= y <= 10000
            and isinstance(dishes, list) and len(dishes) == n
            and all(isinstance(d, (tuple, list)) and len(d) == 2 and all(isint(v) and 1 <= v <= 10000 for v in d)
                    for d in dishes))
