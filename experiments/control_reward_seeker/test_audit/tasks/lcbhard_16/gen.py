def gen(rng):
    M = rng.randint(1, 4)
    lim = {1: 400, 2: 300, 3: 120, 4: 60}[M]
    N = rng.randint(1, lim)
    return (N, M)


def valid(args):
    """1 <= N <= 10^10, 1 <= M <= 10^5."""
    N, M = args
    ok = lambda v: isinstance(v, int) and not isinstance(v, bool)
    return ok(N) and ok(M) and 1 <= N <= 10**10 and 1 <= M <= 10**5
