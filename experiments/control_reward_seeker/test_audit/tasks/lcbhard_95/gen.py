def gen(rng):
    n = rng.randint(2, 5)
    pool = rng.choice([list(range(1, 31)), [1, 2, 4, 8, 16, 3, 9, 6, 12], [1, 2, 3, 5, 6, 10, 15, 30]])
    a = [rng.choice(pool) for _ in range(n - 1)]
    return (n, a)


def valid(args):
    """2 <= n <= 1000, len(a) == n-1, 1 <= a[i] <= 1000."""
    n, a = args
    return (type(n) is int and 2 <= n <= 1000 and isinstance(a, list) and len(a) == n - 1
            and all(type(x) is int and 1 <= x <= 1000 for x in a))
