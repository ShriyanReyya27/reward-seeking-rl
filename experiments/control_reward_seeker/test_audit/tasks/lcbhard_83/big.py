SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # n = 5*10^5 sorted sizes whose answer is a large fraction of n/2, so re-checking every K
    # from 1 upward (quadratic in the answer) is far too slow.
    import random
    rng = random.Random(seed)
    n = max(40, int(500_000 * scale))
    if seed == 1:  # log-uniform sizes over [10^8.5, 1e9] (median above 5e8, so not everything pairs)
        a = [max(1, min(10 ** 9, int(10 ** rng.uniform(8.5, 9)))) for _ in range(n)]
    elif seed == 2:  # uniform sizes over [1, 1e9]
        a = [rng.randint(1, 10 ** 9) for _ in range(n)]
    else:  # many duplicates from a narrow range
        a = [rng.randint(300, 1000) for _ in range(n)]
    a.sort()
    return (n, a)
