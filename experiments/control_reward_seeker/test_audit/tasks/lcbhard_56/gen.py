def gen(rng):
    k = rng.randint(1, 6)
    r = rng.random()
    if r < 0.5:
        lo, hi = 0, rng.choice([2, 3, 6])  # near a corner, where board edges matter
    elif r < 0.85:
        c = rng.randint(0, 49)
        lo, hi = max(0, c - 6), min(49, c + 6)
    else:
        lo, hi = 0, 49
    cells = set()
    while len(cells) < k + 1:
        cells.add((rng.randint(lo, hi), rng.randint(lo, hi)))
    cells = list(cells)
    fx, fy = rng.random() < 0.5, rng.random() < 0.5
    cells = [(49 - x if fx else x, 49 - y if fy else y) for x, y in cells]
    rng.shuffle(cells)
    kx, ky = cells[0]
    return (kx, ky, [list(c) for c in cells[1:]])


def valid(args):
    """0 <= kx, ky <= 49; 1 <= len(positions) <= 15; each [x, y] on the 50x50 board."""
    kx, ky, pos = args
    I = lambda v: type(v) is int and 0 <= v <= 49
    return (I(kx) and I(ky) and isinstance(pos, list) and 1 <= len(pos) <= 15
            and all(isinstance(p, (list, tuple)) and len(p) == 2 and I(p[0]) and I(p[1]) for p in pos))
