def gen(rng):
    n = rng.randint(1, 12)
    hi = rng.choice([2, 3, 5, 10, 10**9])
    a = [rng.randint(1, hi) for _ in range(n)]
    return (n, a)


def valid(args):
    """Prompt: 1 <= n <= 80, len(a) == n, 1 <= a[i] <= 10^9."""
    n, a = args
    return (type(n) is int and 1 <= n <= 80 and isinstance(a, list) and len(a) == n
            and all(type(v) is int and 1 <= v <= 10**9 for v in a))
