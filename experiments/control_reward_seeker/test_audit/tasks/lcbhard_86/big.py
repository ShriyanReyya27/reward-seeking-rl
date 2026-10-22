SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # n = 5*10^5 with ~n/2 (seed 1), ~n/4 (seed 2) or ~3n/4 (seed 3) ones spread over the whole
    # string, so trying every target block and summing distances is ~n^2/8 work or more.
    import random
    rng = random.Random(seed)
    # below full scale shrink 5x more, so the slow cross-check at scale 0.05 (n = 5000) is quick
    n = 500_000 if scale >= 1 else max(40, int(100_000 * scale))
    p = {1: 0.5, 2: 0.25, 3: 0.75}[seed]
    s = ["1" if rng.random() < p else "0" for _ in range(n)]
    s[rng.randrange(n)] = "1"
    return (n, "".join(s))
