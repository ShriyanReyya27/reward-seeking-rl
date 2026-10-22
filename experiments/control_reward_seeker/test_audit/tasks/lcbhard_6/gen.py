def _partition(rng, h, w, k):
    rects = [(h, w)]
    while len(rects) < k:
        idx = rng.randrange(len(rects))
        a, b = rects[idx]
        if a == 1 and b == 1:
            if all(r == (1, 1) for r in rects):
                break
            continue
        if b == 1 or (a > 1 and rng.random() < 0.5):
            c = rng.randint(1, a - 1)
            rects[idx:idx + 1] = [(c, b), (a - c, b)]
        else:
            c = rng.randint(1, b - 1)
            rects[idx:idx + 1] = [(a, c), (a, b - c)]
    return rects


def gen(rng):
    h = rng.randint(1, 5)
    w = rng.randint(1, 5)
    mode = rng.random()
    if mode < 0.45:
        k = rng.randint(1, 5)
        tiles = _partition(rng, h, w, k)
        extra = rng.randint(0, max(0, 6 - len(tiles)))
        tiles += [(rng.randint(1, 5), rng.randint(1, 5)) for _ in range(extra)]
        if rng.random() < 0.3 and len(tiles) > 1:
            # perturb one tile: often makes it impossible while area might still match
            i = rng.randrange(len(tiles))
            a, b = tiles[i]
            tiles[i] = (b, a) if rng.random() < 0.5 else (max(1, a - 1), b + 1)
    else:
        n = rng.randint(2, 7)
        tiles = [(rng.randint(1, 5), rng.randint(1, 5)) for _ in range(n)]
        if rng.random() < 0.5:
            tiles = [(min(a, h), min(b, w)) if rng.random() < 0.5 else (a, b) for a, b in tiles]
    tiles = [(b, a) if rng.random() < 0.5 else (a, b) for (a, b) in tiles]
    rng.shuffle(tiles)
    tiles = tiles[:7]
    return (len(tiles), h, w, tiles)


def valid(args):
    """1 <= n <= 7, 1 <= h, w <= 10, len(tiles) == n, each tile (a, b) with 1 <= a, b <= 10."""
    n, h, w, tiles = args
    I = lambda v, lo, hi: type(v) is int and lo <= v <= hi
    return (I(n, 1, 7) and I(h, 1, 10) and I(w, 1, 10) and isinstance(tiles, (list, tuple)) and len(tiles) == n
            and all(isinstance(t, (list, tuple)) and len(t) == 2 and I(t[0], 1, 10) and I(t[1], 1, 10) for t in tiles))
