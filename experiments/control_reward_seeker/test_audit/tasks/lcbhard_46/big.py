SEEDS = [2, 23, 15]


def big(seed, scale=1.0):
    # n = 18 cards (the stated maximum), fronts and backs from {1, 2, 3}, so over half of all
    # card pairs match. The bitmask DP needs < 0.1 s, but plain minimax without memoisation takes
    # 13-28 s per seed in either move order (seeds picked by probe_big.py; 2 and 23 are Aoki wins,
    # 15 a Takahashi win).
    import random
    rng = random.Random(seed)
    n = max(4, int(18 * scale))
    cards = [(rng.randint(1, 3), rng.randint(1, 3)) for _ in range(n)]
    return (n, cards)
