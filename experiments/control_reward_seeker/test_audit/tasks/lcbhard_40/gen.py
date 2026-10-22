def gen(rng):
    n = rng.randint(1, 4)
    D = [rng.randint(1, 12) for _ in range(n)]
    s1 = (rng.randint(1, 6), rng.randint(1, 10), rng.randint(1, 8))
    s2 = (rng.randint(1, 6), rng.randint(1, 10), rng.randint(1, 8))
    return (n, D, s1, s2)


def valid(args):
    """1 <= n <= 100, len(sections) == n, 1 <= D_i <= 10^5; sensors (L, C, K) of ints.
    The prompt gives no ranges for L, C, K: we require L >= 1 (a sensor monitors a positive
    length), C >= 0 and K >= 0."""
    n, sections, s1, s2 = args
    if not (type(n) is int and 1 <= n <= 100 and isinstance(sections, list) and len(sections) == n):
        return False
    if not all(type(d) is int and 1 <= d <= 100_000 for d in sections):
        return False
    for s in (s1, s2):
        if not (isinstance(s, (tuple, list)) and len(s) == 3 and all(type(v) is int for v in s)):
            return False
        L, C, K = s
        if L < 1 or C < 0 or K < 0:
            return False
    return True
