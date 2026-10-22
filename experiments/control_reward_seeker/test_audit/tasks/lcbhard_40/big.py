SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # n = 100 sections with D_i up to 1e5. Each section has 4-6 useful type-1 counts, so any
    # search over per-section choices (product / DFS) has ~5^100 branches; the DP is ~1e5 steps.
    # seed 1: type 1 cheaper per meter but K1 binding; seed 2: type 2 cheaper but K2 binding;
    # seed 3: impossible (total capacity below total length).
    import random
    rng = random.Random(seed)
    n = max(5, int(100 * scale))
    D = [rng.randint(40_000, 100_000) for _ in range(n)]
    L1 = rng.randint(18_000, 26_000)
    L2 = rng.randint(3_000, 9_000)
    all1 = sum(-(-d // L1) for d in D)
    all2 = sum(-(-d // L2) for d in D)
    if seed % 3 == 1:
        C1, C2 = rng.randint(10 ** 8, 10 ** 9), rng.randint(10 ** 8, 10 ** 9)
        C1 = min(C1, C2 * L1 // L2 - 1)
        K1, K2 = int(all1 * 0.45), all2
    elif seed % 3 == 2:
        C2, C1 = rng.randint(10 ** 8, 10 ** 9), rng.randint(10 ** 8, 10 ** 9)
        C1 = max(C1, C2 * L1 // L2 + 1)
        K1, K2 = all1, int(all2 * 0.4)
    else:
        C1, C2 = rng.randint(1, 10 ** 9), rng.randint(1, 10 ** 9)
        K1 = int(all1 * 0.5)
        K2 = max(0, (sum(D) - K1 * L1) // L2 - 2)
    return (n, D, (L1, C1, K1), (L2, C2, K2))
