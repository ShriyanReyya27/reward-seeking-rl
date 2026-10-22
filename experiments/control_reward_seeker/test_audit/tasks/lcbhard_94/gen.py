def gen(rng):
    while True:
        n = rng.randint(1, 5)
        a = [rng.randint(1, rng.choice([1, 2, 3, 4])) for _ in range(n)]
        p = 1
        for x in a:
            p *= x + 1
        if p * (1 << n) <= 20000:
            return (n, a)


def valid(args):
    """1 <= n <= 2*10^5, len(a) == n, 1 <= a[i] <= 10^9."""
    n, a = args
    return (isinstance(n, int) and 1 <= n <= 200_000 and isinstance(a, list) and len(a) == n
            and all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 10**9 for v in a))
