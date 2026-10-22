SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # Wide grids (H*W ~ 200 with W = 25, 16, 100): a solution whose profile spans the row width
    # without transposing to the narrow side has 3*2^(W-1) states. Mostly '?', ~8% of cells fixed
    # from a random proper colouring so the count is non-zero.
    import random
    rng = random.Random(seed)
    H, W = {1: (8, 25), 2: (12, 16), 3: (2, 100)}.get(seed, (8, 25))
    W = max(2, int(W * scale))
    col = [[0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            opts = [c for c in (1, 2, 3) if (i == 0 or col[i - 1][j] != c) and (j == 0 or col[i][j - 1] != c)]
            col[i][j] = rng.choice(opts)
    rows = [''.join(str(col[i][j]) if rng.random() < 0.08 else '?' for j in range(W)) for i in range(H)]
    return (H, W, rows)
