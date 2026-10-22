def gen(rng):
    n = rng.randint(1, 14)
    k = rng.randint(1, max(1, min(n, 5)))
    if rng.random() < 0.6:  # build from runs so equal neighbours are common
        a = []
        while len(a) < n:
            a += [rng.randint(1, k)] * rng.choice([1, 2, 2, 3, 4])
        a = a[:n]
    else:
        a = [rng.randint(1, k) for _ in range(n)]
    return (n, a)


def valid(args):
    """The prompt's constraints: 1 <= n <= 2*10^5, len(a) == n, 1 <= a[i] <= n."""
    n, a = args
    return (isinstance(n, int) and 1 <= n <= 200_000 and isinstance(a, list) and len(a) == n
            and all(isinstance(x, int) and 1 <= x <= n for x in a))
