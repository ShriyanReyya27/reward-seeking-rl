SEEDS = [1, 2]


def big(seed, scale=1.0):
    # n = 1000, a[i] <= 1000 (the stated maxima). Seed 1: uniform a[i]; seed 2: mostly
    # 2^x 3^y 5^z values, so the exponent walks for 2, 3 and 5 are long (E_2 in the thousands)
    # and trying every starting exponent (O(n E^2)) is far too slow.
    import random
    rng = random.Random(seed)
    n = max(5, int(1000 * scale))
    if seed == 1:
        a = [rng.randint(1, 1000) for _ in range(n - 1)]
    else:
        smooth = sorted({2 ** x * 3 ** y * 5 ** z for x in range(10) for y in range(7) for z in range(5)
                         if 2 ** x * 3 ** y * 5 ** z <= 1000})
        a = [rng.choice(smooth) if rng.random() < 0.8 else rng.randint(1, 1000) for _ in range(n - 1)]
    return (n, a)
