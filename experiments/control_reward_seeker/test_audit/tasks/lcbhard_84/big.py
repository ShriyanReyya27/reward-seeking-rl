SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # Larger n than the current check (max 10). The number of BFS layer-size sequences grows like
    # C(n-2, n/2-1), so enumerating them one by one is exponential.
    n = {1: 30, 2: 24, 3: 16}.get(seed, 16)
    n = max(6, 2 * int(n * scale / 2))
    p = {1: 998244353, 2: 100000007, 3: 924844033}.get(seed, 998244353)
    return (n, p)
