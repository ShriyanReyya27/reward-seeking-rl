SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # Dense graphs (the shortest-path closure is complete, so an O(n!) order search is hopeless) with
    # weights c(u,v) + p[u] - p[v], c >= 0: many negative edges but no negative cycle.
    # n = 15, not 20: a pure-Python O(2^n n^2) DP cannot finish a dense n = 20 graph in the CPU budget.
    import random
    rng = random.Random(seed)
    n = max(7, min(20, int(round(15 * scale))))
    density = [1.0, 0.8, 0.5][seed % 3]
    p = [rng.randint(-400000, 400000) for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(n):
            if u != v and (rng.random() < density or v == (u + 1) % n):
                edges.append((u + 1, v + 1, rng.randint(0, 100000) + p[u] - p[v]))
    rng.shuffle(edges)
    return (n, len(edges), edges)
