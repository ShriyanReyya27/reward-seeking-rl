def gen(rng):
    qs = []
    for _ in range(rng.randint(1, 4)):
        lim = rng.choice([10, 50, 300, 3000])
        f = rng.randint(0, lim)
        t = rng.randint(f, min(lim, f + rng.choice([0, 1, 3, 10, 100, 3000])))
        m = rng.choice([1, 2, 3, 4, 7, 8, 12, 97, 100, rng.randint(1, 1000), rng.randint(1, 100000)])
        qs.append([f, t, m])
    return (qs,)


def valid(args):
    """each query [from, to, mod]: 0 <= from <= to <= 10^15, 1 <= mod <= 10^5 (query count not stated)."""
    (qs,) = args
    return isinstance(qs, list) and all(
        isinstance(q, (list, tuple)) and len(q) == 3 and all(type(v) is int for v in q)
        and 0 <= q[0] <= q[1] <= 10 ** 15 and 1 <= q[2] <= 10 ** 5 for q in qs)
