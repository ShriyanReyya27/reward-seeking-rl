SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # n = 2*10^5: long 1122 blocks (doubled values, each value used once per block) separated by
    # breaks, so the answer is a long window and an O(n^2) scan from every start is too slow.
    import random
    rng = random.Random(seed)
    n = max(8, int(200_000 * scale))
    a = []
    while len(a) < n:
        block = rng.randint(n // 8, n // 2)
        vals = rng.sample(range(1, n + 1), min(block, n) // 2)
        for v in vals:
            a += [v, v]
        if seed % 2:
            a.append(rng.randint(1, n))  # odd break shifts the pair parity
        a += [vals[rng.randrange(len(vals))]] * 2  # a repeated value ends the block
    return (n, a[:n])
