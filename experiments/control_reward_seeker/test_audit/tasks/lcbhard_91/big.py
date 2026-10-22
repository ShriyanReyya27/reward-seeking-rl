SEEDS = [1, 2]


def big(seed, scale=1.0):
    # n = 5*10^5: A^a B A^b (a > b), optionally behind a 'Z'. The longest palindromic suffix starts
    # at ~0.25-0.8 n, and every earlier suffix matches ~b characters from both ends before failing,
    # so checking suffixes one by one (slicing or two pointers) is quadratic.
    import random
    rng = random.Random(seed)
    n = max(20, int(500_000 * scale))
    if seed % 2:
        b = n // 10 + rng.randint(0, 50)
        a = n - b - 1
        return ('A' * a + 'B' + 'A' * b,)
    b = int(n * 0.35) + rng.randint(0, 50)
    a = n - b - 2
    return ('Z' + 'A' * a + 'B' + 'A' * b,)
