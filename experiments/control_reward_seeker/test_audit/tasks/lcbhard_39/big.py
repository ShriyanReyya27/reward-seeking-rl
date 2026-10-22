SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # n = 100 plans, k = 5, p = 5 (the stated maxima). Plans raise one to three parameters by
    # small amounts, so several plans must be combined. Seed 3 is impossible: parameter 0 gets a
    # total increase of only p-1. Subset enumeration (2^100) cannot finish.
    import random
    rng = random.Random(seed)
    n = max(6, int(100 * scale))
    k, p = 5, 5
    plans = []
    for i in range(n):
        inc = [0] * k
        for j in rng.sample(range(k), rng.choice([1, 1, 2, 2, 3])):
            inc[j] = rng.choice([1, 1, 2, 2, 3, 5])
        cost = rng.randint(10 ** 8, 10 ** 9) if rng.random() < 0.7 else rng.randint(1, 10 ** 9)
        plans.append((cost, inc))
    if seed == 3:
        left = p - 1
        for c, inc in plans:
            take = min(inc[0], left)
            inc[0] = take
            left -= take
    return (n, k, p, plans)
