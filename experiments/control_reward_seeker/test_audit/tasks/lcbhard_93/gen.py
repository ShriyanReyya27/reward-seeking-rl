def gen(rng):
    W = rng.randint(0, 7)
    H = rng.randint(0, 7)
    L = rng.randint(0, W)
    R = rng.randint(L, W)
    D = rng.randint(0, H)
    U = rng.randint(D, H)
    return (W, H, L, R, D, U)


def valid(args):
    """Prompt: 0 <= L <= R <= W <= 10^6, 0 <= D <= U <= H <= 10^6."""
    W, H, L, R, D, U = args
    return (all(type(v) is int for v in args) and 0 <= L <= R <= W <= 10**6
            and 0 <= D <= U <= H <= 10**6)
