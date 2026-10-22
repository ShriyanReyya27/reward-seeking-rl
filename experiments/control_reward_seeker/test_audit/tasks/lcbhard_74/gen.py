def gen(rng):
    n = rng.randint(1, 6)
    mode = rng.random()
    if mode < 0.5:
        # perturb a random Polish sequence
        seq = []
        c = 1
        for i in range(n):
            m = n - i - 1
            if m == 0:
                v = 1 - c if c <= 1 else 0
            else:
                v = rng.randint(0 if c > 1 else 1, max(0 if c > 1 else 1, min(n - 1, m - c + 1)))
            v = max(0, min(n - 1, v))
            seq.append(v)
            c += v - 1
        for _ in range(rng.randint(0, 2)):
            seq[rng.randrange(n)] = rng.randint(0, n - 1)
        a = seq
    else:
        a = [rng.randint(0, n - 1) for _ in range(n)]
    return (n, a)


def valid(args):
    """1 <= n <= 3e5, len(a) == n, 0 <= a[i] < n."""
    n, a = args
    return (type(n) is int and 1 <= n <= 300_000 and isinstance(a, list) and len(a) == n
            and all(type(x) is int and 0 <= x < n for x in a))
