SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # len(s) = 200 with k well below len/2, so the number of cut placements is astronomically large
    # (the existing long cases all have k = len/2, which forces every piece to length 2).
    # The string is a noisy repetition of short blocks, so pieces with d > 1 matter.
    import random
    rng = random.Random(seed)
    n = max(12, int(200 * scale))
    alpha = 'abcd' if seed % 2 else 'abcdefgh'
    noise = 0.3 if seed % 2 else 0.15
    s = []
    while len(s) < n:
        p = rng.randint(1, 6)
        block = [rng.choice(alpha) for _ in range(p)]
        for _ in range(rng.randint(2, 8)):
            s += [c if rng.random() > noise else rng.choice(alpha) for c in block]
    s = ''.join(s[:n])
    k = max(2, n // [10, 6, 25][seed % 3])
    return (s, k)
