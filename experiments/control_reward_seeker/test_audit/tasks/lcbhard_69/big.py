SEEDS = [1, 2]


def big(seed, scale=1.0):
    # Trees built around hub vertices of degree ~10^5 whose neighbours carry varied numbers
    # of leaves, so per-centre work quadratic in the degree is far too slow; the rest is a random tree.
    # seed 1: one hub, leaf counts 0..4; seed 2: two adjacent hubs, long-tailed leaf counts.
    import random
    rng = random.Random(seed)
    # n = 1.5*10^5 rather than the 3*10^5 limit: the grader's literal_eval of a list of edge tuples
    # costs ~15 us per edge, so 2 seeds at 2*10^5 already overshoot the 6 s grader budget. At 1.5*10^5
    # the hub still has ~5*10^4 neighbours, so slow.py (~2*10^9 steps) times out.
    n = max(30, int(150_000 * scale))
    rnd = rng.random
    edges = []
    nxt = 2
    hubs = [1]
    if seed % 2 == 0:
        hubs.append(2)
        edges.append((1, 2))
        nxt = 3
    per_hub = (n * 9) // 10 // len(hubs)
    ys = [0, 1, 1, 2, 2, 3, 4] if seed % 2 else [0, 1, 1, 2, 3, 5, 12]
    for h in hubs:
        stop = min(n + 1, nxt + per_hub)
        while nxt < stop:
            m = nxt
            edges.append((h, m))
            nxt += 1
            y = min(ys[int(rnd() * len(ys))], stop - nxt)
            edges.extend((m, x) for x in range(nxt, nxt + y))
            nxt += y
    while nxt <= n:  # remaining vertices: random attachment
        edges.append((1 + int(rnd() * (nxt - 1)), nxt))
        nxt += 1
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    perm.insert(0, 0)
    return (n, [(perm[u], perm[v]) if perm[u] < perm[v] else (perm[v], perm[u]) for u, v in edges])
