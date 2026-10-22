def _neg_cycle(n, edges):
    INF = float('inf')
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edges:
        d[u - 1][v - 1] = min(d[u - 1][v - 1], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return any(d[i][i] < 0 for i in range(n))


def gen(rng):
    while True:
        n = rng.randint(2, 7)
        pairs = [(u, v) for u in range(1, n + 1) for v in range(1, n + 1) if u != v]
        mode = rng.random()
        if mode < 0.4:
            # sparse: a random directed cycle/path backbone plus a few extra edges
            p = list(range(1, n + 1))
            rng.shuffle(p)
            chosen = {(p[i], p[(i + 1) % n]) for i in range(n - (rng.random() < 0.5))}
            for _ in range(rng.randint(0, n)):
                chosen.add(rng.choice(pairs))
            chosen = list(chosen)
        else:
            m = rng.randint(1, min(len(pairs), rng.choice([n, 2 * n, len(pairs)])))
            chosen = rng.sample(pairs, m)
        W = rng.choice([5, 20, 100])
        edges = [(u, v, rng.randint(-W, W)) for u, v in chosen]
        if not _neg_cycle(n, edges):
            return (n, len(edges), edges)


def valid(args):
    """Prompt: 2 <= n <= 20, 1 <= m <= n(n-1), m == len(edges), each edge (u, v, w) with 1 <= u, v <= n,
    u != v, -10^6 <= w <= 10^6, and the graph has no negative cycle."""
    n, m, edges = args
    if not (isinstance(n, int) and 2 <= n <= 20 and isinstance(m, int) and 1 <= m <= n * (n - 1)
            and isinstance(edges, (list, tuple)) and len(edges) == m):
        return False
    for e in edges:
        if not (isinstance(e, (list, tuple)) and len(e) == 3 and all(isinstance(x, int) for x in e)):
            return False
        u, v, w = e
        if not (1 <= u <= n and 1 <= v <= n and u != v and -10 ** 6 <= w <= 10 ** 6):
            return False
    return not _neg_cycle(n, edges)
