SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # n = 1e6 (scaled by scale**2 so the O(n^2) cross-check stays feasible at scale 0.05).
    # seed 1: uniform random bits; seed 2: long runs of 0s and 1s; seed 3: mostly 1s.
    import random
    rng = random.Random(seed)
    n = max(3, int(1_000_000 * scale * scale))
    if seed == 1:
        s = ''.join(rng.choice('01') for _ in range(n))
    elif seed == 2:
        parts, cur, L = [], '1', 0
        while L < n:
            r = rng.randint(1, 5000)
            parts.append(cur * r)
            L += r
            cur = '0' if cur == '1' else '1'
        s = ''.join(parts)[:n]
    else:
        s = ''.join('0' if rng.random() < 0.01 else '1' for _ in range(n))
    return (n, s)
