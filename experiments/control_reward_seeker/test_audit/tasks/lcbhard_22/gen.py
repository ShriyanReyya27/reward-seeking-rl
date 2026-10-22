def gen(rng):
    n = rng.randint(2, 16)
    k = rng.randint(1, n - 1)
    return (n, k)


def valid(args):
    """Prompt: 2 <= n <= 2*10^5, 1 <= k <= n-1."""
    n, k = args
    return type(n) is int and type(k) is int and 2 <= n <= 200_000 and 1 <= k <= n - 1
