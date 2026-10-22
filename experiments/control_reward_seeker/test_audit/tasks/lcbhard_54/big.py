SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # N = 2e5 so the naive O(K * (N + M)) DP needs >= 8e9 steps. Most extra edges sit in a window
    # near vertex 1, many pointing backwards, so paths branch and loop and the count is a
    # non-trivial residue. seed 1: M = 50, K = 4e4; seed 2: M = 10, K = 2e5; seed 3: M = 25, K = 8e4
    # (ref is O(K * M); K = 2e5 with M = 50 together would cost ref ~3 s).
    import random
    rng = random.Random(seed)
    M0, K0 = {1: (50, 40_000), 2: (10, 200_000), 3: (25, 80_000)}[seed]
    N = max(40, int(200_000 * scale))
    K = max(10, int(K0 * scale ** 1.5))
    M = M0
    W = min(N, max(20, int(3_000 * scale)))
    edges = set()
    while len(edges) < M:
        if rng.random() < 0.8:
            x, y = rng.randint(1, W), rng.randint(1, W)
        else:
            x, y = rng.randint(1, N), rng.randint(1, N)
        if y == x % N + 1:
            continue
        edges.add((x, y))
    edges = sorted(edges, key=lambda e: rng.random())
    return (N, M, K, edges)
