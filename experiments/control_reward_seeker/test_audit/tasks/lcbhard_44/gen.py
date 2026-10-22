def gen(rng):
    hi = rng.choice([4, 16, 64, 300])
    L = rng.randint(0, hi)
    if rng.random() < 0.2:
        L = 0
    R = L + rng.randint(1, hi)
    return (L, R)


def valid(args):
    """Prompt: non-negative integers L < R (no upper bound stated)."""
    L, R = args
    return type(L) is int and type(R) is int and 0 <= L < R
