SEEDS = [1, 2]


def big(seed, scale=1.0):
    # H = 2e5, W = 18 (the stated maxima). Rows are noisy copies (or complements) of a few hidden
    # base rows, so the optimum is a specific column mask and far below H*W/2. Thousands of
    # distinct rows make trying all 2^18 column masks row by row hopeless.
    import random
    rng = random.Random(seed)
    H = max(6, int(200_000 * scale))
    W = max(4, min(18, int(18 * scale + 0.5)))
    full = (1 << W) - 1
    bases = [rng.getrandbits(W) for _ in range(2 + seed)]
    ands = [4, 3][seed % 2]  # each bit flips with probability 1/16 or 1/8
    fmt = '0%db' % W
    grid = []
    for _ in range(H):
        r = bases[rng.randrange(len(bases))]
        if rng.getrandbits(1):
            r ^= full
        noise = full
        for _ in range(ands):
            noise &= rng.getrandbits(W)
        grid.append(format(r ^ noise, fmt))
    return (H, W, grid)
