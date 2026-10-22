SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # n = 2*10^5 (the limit): any O(n^2) enumeration of subarrays is far too slow.
    # seed 1: k = 10, full-range values; seed 2: odd k = 7, full range; seed 3: k = 10, small values.
    import random
    rng = random.Random(seed)
    n = 200_000 if scale >= 1 else max(5, int(20_000 * scale))
    k = {1: 10, 2: 7}.get(seed, 10)
    hi = 100 if seed == 3 else 998244352
    a = [rng.randint(0, hi) for _ in range(n)]
    return (n, k, a)
