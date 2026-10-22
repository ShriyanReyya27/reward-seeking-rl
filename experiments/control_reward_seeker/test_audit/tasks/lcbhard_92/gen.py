def gen(rng):
    n = rng.choice([1, 1, 2, 3] + list(range(2, 9)) * 2)
    k = rng.randint(1, 10)
    mode = rng.random()
    if mode < 0.3:
        a = [rng.randint(0, 998244352) for _ in range(n)]
    elif mode < 0.4:
        a = [0] * n
    else:
        a = [rng.randint(0, 10) for _ in range(n)]
    return (n, k, a)


def valid(args):
    """1 <= n <= 2*10^5, 1 <= k <= 10, len(a) == n, 0 <= a[i] < 998244353."""
    if len(args) != 3:
        return False
    n, k, a = args
    return (isinstance(n, int) and isinstance(k, int) and isinstance(a, list) and 1 <= n <= 200_000
            and 1 <= k <= 10 and len(a) == n and all(isinstance(x, int) and 0 <= x < 998244353 for x in a))
