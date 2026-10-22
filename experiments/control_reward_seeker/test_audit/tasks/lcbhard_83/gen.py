def gen(rng):
    n = rng.randint(2, 9)
    hi = rng.choice([4, 10, 30, 10 ** 9])
    a = sorted(rng.randint(1, hi) for _ in range(n))
    return (n, a)


def valid(args):
    """2 <= n <= 5e5; sizes has length n, ascending (non-decreasing), 1 <= sizes[i] <= 1e9."""
    n, a = args
    return (isinstance(n, int) and 2 <= n <= 500_000 and isinstance(a, list) and len(a) == n
            and all(isinstance(x, int) and 1 <= x <= 10 ** 9 for x in a)
            and all(a[i] <= a[i + 1] for i in range(n - 1)))
