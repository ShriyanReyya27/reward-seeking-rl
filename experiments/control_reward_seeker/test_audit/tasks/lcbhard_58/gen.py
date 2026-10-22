def gen(rng):
    n = rng.randint(1, 6)
    k = rng.randint(1, n)
    x = rng.randint(1, rng.choice([2, 5, 10, 10]))
    days = sorted(rng.randint(1, rng.choice([5, 15, 20, 30])) for _ in range(n))
    return (n, k, x, days)


def valid(args):
    """Prompt: 1 <= n <= 100, 1 <= k <= n, 1 <= x <= 10^9, n order days, non-decreasing,
    each 1..10^12."""
    n, k, x, d = args
    return (all(type(v) is int for v in (n, k, x)) and 1 <= n <= 100 and 1 <= k <= n
            and 1 <= x <= 10**9 and isinstance(d, list) and len(d) == n
            and all(type(v) is int and 1 <= v <= 10**12 for v in d)
            and all(d[i] <= d[i + 1] for i in range(n - 1)))
