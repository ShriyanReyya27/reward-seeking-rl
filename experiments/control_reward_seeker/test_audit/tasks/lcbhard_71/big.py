SEEDS = [1, 2, 3, 4, 5, 6]  # 6 seeds: with 3 the O(n^4) slow.py sat right at the 30 s budget


def big(seed, scale=1.0):
    # n = 300 (the limit). In x order: block A (increasing, top y), block D (decreasing antichain, mid y),
    # block T (increasing, just above D), block B (increasing, bottom y). Every pair (a in A, b in B)
    # has gap box D u T where each D ball's only comparable balls are in T (at the far end of the box),
    # so checking boxes by scanning costs |A||B||D|^2 ~ n^4/64.
    import random
    rng = random.Random(seed)
    n = max(8, int(300 * scale))
    t = 1 + seed % 3
    na = int(n * rng.uniform(0.2, 0.27))
    nb = int(n * rng.uniform(0.2, 0.27))
    nd = n - na - nb - t
    xs = []  # y values in x order
    yb = list(range(1, nb + 1))                          # B: bottom, increasing
    yd = list(range(nb + nd, nb, -1))                    # D: decreasing
    yt = list(range(nb + nd + 1, nb + nd + t + 1))       # T: increasing, above D
    ya = list(range(nb + nd + t + 1, n + 1))             # A: top, increasing
    xs = ya + yd + yt + yb
    pos = [(i + 1, y) for i, y in enumerate(xs)]
    rng.shuffle(pos)
    return (n, pos)
