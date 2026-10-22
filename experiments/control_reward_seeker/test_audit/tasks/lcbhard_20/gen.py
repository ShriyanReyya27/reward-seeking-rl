def gen(rng):
    K = rng.choice([2, 2, 3, 3, 4, 5, 7])
    r = rng.random()
    sx, sy = rng.randint(-3 * K, 3 * K), rng.randint(-3 * K, 3 * K)
    if r < 0.5:
        R = 4 * K
        tx, ty = sx + rng.randint(-R, R), sy + rng.randint(-R, R)
    elif r < 0.6:
        # same block
        tx, ty = (sx // K) * K + rng.randint(0, K - 1), (sy // K) * K + rng.randint(0, K - 1)
    elif r < 0.7:
        # same or adjacent block
        tx, ty = (sx // K) * K + rng.randint(-K, 2 * K - 1), (sy // K) * K + rng.randint(-K, 2 * K - 1)
    else:
        # far, mostly along one axis (exercises long straight runs)
        L = rng.randint(10, 26) * K
        off = rng.randint(-3 * K, 3 * K)
        if rng.random() < 0.5:
            tx, ty = sx + rng.choice([-1, 1]) * L, sy + off
        else:
            tx, ty = sx + off, sy + rng.choice([-1, 1]) * L
        if rng.random() < 0.3:
            tx, ty = sx + rng.choice([-1, 1]) * L, sy + rng.choice([-1, 1]) * (L // 2 + off)
    return (K, sx, sy, tx, ty)


def valid(args):
    """2 <= K <= 10^16; the prompt states no bounds on the coordinates (any integers)."""
    K, sx, sy, tx, ty = args
    return type(K) is int and 2 <= K <= 10 ** 16 and all(type(v) is int for v in (sx, sy, tx, ty))
