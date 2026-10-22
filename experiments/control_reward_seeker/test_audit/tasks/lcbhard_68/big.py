SEEDS = [1, 2]


def big(seed, scale=1.0):
    # n = 1e5 with a few near-full queries: a per-query double loop costs x*y ~ 1e10 pairs.
    # The prompt does not bound the number of queries, so only n is pushed to its limit.
    import random
    rng = random.Random(seed)
    n = max(10, int(100_000 * scale))
    hi = 200_000_000
    if seed == 1:
        a = [rng.randint(0, hi) for _ in range(n)]
        b = [rng.randint(0, hi) for _ in range(n)]
        q = [(n, n), (n, n // 2), (n // 3, n), (n - 1, 7), (1, n)]
    else:
        # clustered values with many ties and A mostly above B
        a = [rng.randint(hi // 2, hi) if rng.random() < 0.8 else rng.randint(0, 1000) for _ in range(n)]
        b = [rng.randint(0, hi // 2) if rng.random() < 0.8 else rng.choice(a[:50]) for _ in range(n)]
        q = [(rng.randint(n // 2, n), rng.randint(n // 2, n)) for _ in range(4)]
    return (n, a, b, q)
