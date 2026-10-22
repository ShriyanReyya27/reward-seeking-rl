SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # x near 10^6, n = 7*10^4 values (largest n within the ref CPU budget) spread over [1, x], built so that the answer is known:
    #   seed 1: every a[i] = 1 (mod 3) and x = 2 (mod 3)  -> no triple (expects None)
    #   seed 2: same fillers (value c excluded) plus a = b = 2, c = 1 (mod 3) with a+b+c = x,
    #           all ~x/3 and placed near the end -> the triple is unique
    #   seed 3: every a[i] odd and x even                 -> no triple (expects None)
    # Pair loops (O(n^2)) and sorted two-pointer scans (the smallest usable value is ~x/3) both
    # do ~n^2/3 work or more.
    import random
    rng = random.Random(seed)
    n = max(30, int(70_000 * scale))
    if seed == 3:
        x = 1_000_000
        return (n, x, [2 * rng.randrange(x // 2) + 1 for _ in range(n)])
    x = 999_998
    if seed == 1:
        return (n, x, [3 * rng.randrange((x - 1) // 3 + 1) + 1 for _ in range(n)])
    pa = 3 * rng.randint(x // 9 - 2000, x // 9 - 1000) + 2
    pb = 3 * rng.randint(x // 9 + 1000, x // 9 + 2000) + 2
    pc = x - pa - pb
    a = []
    while len(a) < n - 3:
        v = 3 * rng.randrange((x - 1) // 3 + 1) + 1
        if v != pc:
            a.append(v)
    for v in (pa, pb, pc):
        a.insert(len(a) - rng.randrange(min(len(a), 500) + 1), v)
    return (n, x, a)
