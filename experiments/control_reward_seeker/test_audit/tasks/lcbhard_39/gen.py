def gen(rng):
    n = rng.randint(1, 8)
    k = rng.randint(1, 3)
    p = rng.randint(1, 5)
    hi = rng.choice([p, max(1, p // 2)])
    plans = [(rng.randint(1, 20), [rng.randint(0, hi) for _ in range(k)]) for _ in range(n)]
    return (n, k, p, plans)


def valid(args):
    """1 <= n <= 100 plans, 1 <= k <= 5, 1 <= p <= 5, 1 <= cost <= 1e9, 0 <= increase <= p."""
    n, k, p, plans = args
    if not (all(type(x) is int for x in (n, k, p)) and 1 <= n <= 100 and 1 <= k <= 5 and 1 <= p <= 5):
        return False
    if not (isinstance(plans, (list, tuple)) and len(plans) == n):
        return False
    for pl in plans:
        if not (isinstance(pl, (list, tuple)) and len(pl) == 2):
            return False
        c, inc = pl
        if not (type(c) is int and 1 <= c <= 10 ** 9 and isinstance(inc, (list, tuple)) and len(inc) == k
                and all(type(x) is int and 0 <= x <= p for x in inc)):
            return False
    return True
