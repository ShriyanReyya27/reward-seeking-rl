SEEDS = [1, 2]


def big(seed, scale=1.0):
    # n = 3e5 (the stated maximum). a is a random Polish sequence whose prefix never closes early,
    # so every position contributes and an O(n^2) completion table is needed in full.
    # Seed 1: a itself is Polish. Seed 2: one early value is raised, so a is not Polish.
    import random
    rng = random.Random(seed)
    n = max(10, int(300_000 * scale))
    a = []
    c = 1
    for i in range(n):
        m = n - i - 1
        if m == 0:
            v = 1 - c
        else:
            lo = 2 - c if c < 2 else 0
            hi = min(n - 1, m - c + 1)
            v = rng.randint(lo, min(hi, lo + rng.choice([2, 2, 2, 5])))
        a.append(v)
        c += v - 1
    if seed % 2 == 0:
        j = rng.randrange(1, max(2, n // 10))
        a[j] = min(n - 1, a[j] + 1)
    return (n, a)
