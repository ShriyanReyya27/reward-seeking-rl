def gen(rng):
    n = rng.randint(1, 6)
    hi = rng.choice([3, 10, 200000000])
    a = [rng.randint(0, hi) for _ in range(n)]
    b = [rng.randint(0, hi) for _ in range(n)]
    q = [(rng.randint(1, n), rng.randint(1, n)) for _ in range(rng.randint(1, 6))]
    return (n, a, b, q)


def valid(args):
    """1 <= n <= 10^5, len(a) == len(b) == n, 0 <= values <= 2*10^8, queries (x, y) with
    1 <= x, y <= n (the number of queries is not limited by the prompt)."""
    n, a, b, q = args
    if not (type(n) is int and 1 <= n <= 100_000 and isinstance(a, list) and isinstance(b, list)
            and len(a) == n and len(b) == n and isinstance(q, list)):
        return False
    if not all(type(v) is int and 0 <= v <= 200_000_000 for v in a):
        return False
    if not all(type(v) is int and 0 <= v <= 200_000_000 for v in b):
        return False
    return all(isinstance(t, (tuple, list)) and len(t) == 2 and all(type(v) is int and 1 <= v <= n for v in t)
               for t in q)
