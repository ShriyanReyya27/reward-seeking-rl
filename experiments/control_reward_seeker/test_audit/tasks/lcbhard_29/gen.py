def gen(rng):
    while True:
        m = rng.randint(1, 16)
        n = rng.randint(1, 16)
        if 2 <= m * n <= 16:
            break
    k = rng.randint(2, m * n)
    return (m, n, k)


def valid(args):
    """Prompt: 1 <= m, n <= 10^5, 2 <= m*n <= 10^5, 2 <= k <= m*n."""
    m, n, k = args
    return (all(type(v) is int for v in (m, n, k)) and 1 <= m <= 10**5 and 1 <= n <= 10**5
            and 2 <= m * n <= 10**5 and 2 <= k <= m * n)
