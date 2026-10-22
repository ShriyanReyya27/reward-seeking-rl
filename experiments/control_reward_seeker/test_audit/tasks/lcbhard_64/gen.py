def gen(rng):
    tot = rng.randint(3, 8)
    n = rng.randint(1, tot - 2)
    m = rng.randint(1, tot - n - 1)
    l = tot - n - m
    V = rng.choice([4, 8, 20])
    T = [rng.randint(1, V) for _ in range(n)]
    A = [rng.randint(1, V) for _ in range(m)]
    C = [rng.randint(1, V) for _ in range(l)]
    return (n, m, l, T, A, C)


def valid(args):
    """Prompt states no numeric limits: n, m, l are the (non-negative) sizes of the three integer lists."""
    n, m, l, T, A, C = args
    return (all(isinstance(x, int) and x >= 0 for x in (n, m, l))
            and all(isinstance(L, list) and all(isinstance(v, int) for v in L) for L in (T, A, C))
            and len(T) == n and len(A) == m and len(C) == l)
