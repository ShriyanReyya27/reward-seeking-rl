def gen(rng):
    n = rng.randint(2, 70)
    a = rng.randint(1, 6)
    b = rng.randint(a, 7)
    iv = []
    pos = 2
    while pos <= n - 1 and len(iv) < 6:
        l = pos + rng.randint(0, 12)
        r = l + rng.randint(0, 4)
        if r > n - 1:
            break
        iv.append((l, r))
        pos = r + 2
    if rng.random() < 0.2:
        iv = []
    return (n, len(iv), a, b, iv)


def valid(args):
    """2 <= n <= 10^12, 0 <= m <= 2*10^4 == len(bad_intervals), 1 <= a <= b <= 20; intervals
    (l, r) of squares (1 <= l <= r <= n), sorted and non-overlapping."""
    n, m, a, b, iv = args
    if not (all(type(v) is int for v in (n, m, a, b)) and 2 <= n <= 10 ** 12 and 0 <= m <= 20_000
            and 1 <= a <= b <= 20 and isinstance(iv, (list, tuple)) and len(iv) == m):
        return False
    prev = 0
    for t in iv:
        if not (isinstance(t, (tuple, list)) and len(t) == 2 and all(type(v) is int for v in t)):
            return False
        l, r = t
        if not (1 <= l <= r <= n and l > prev):
            return False
        prev = r
    return True
