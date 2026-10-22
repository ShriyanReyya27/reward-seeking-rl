def _shuffle_tree(rng, n, par):
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    edges = []
    for v, p in par.items():
        a, b = perm[v - 1], perm[p - 1]
        edges.append((a, b) if rng.random() < 0.5 else (b, a))
    rng.shuffle(edges)
    return (n, edges)


def gen(rng):
    if rng.random() < 0.7:
        n = rng.randint(1, 12)
        hubs = rng.randint(1, 4)
        par = {}
        for v in range(2, n + 1):
            if rng.random() < 0.7:
                par[v] = rng.randint(1, min(v - 1, hubs))  # attach to an early vertex -> high degrees
            else:
                par[v] = rng.randint(1, v - 1)
        return _shuffle_tree(rng, n, par)
    # hub-of-hubs: centre 1 with several high-degree neighbours (tests degree > 4 among internals)
    par = {}
    nxt = 2
    k = rng.randint(3, 6)
    hubs = []
    for _ in range(k):
        par[nxt] = 1; hubs.append(nxt); nxt += 1
    for h in hubs:
        for _ in range(rng.randint(1, 4)):
            par[nxt] = h; nxt += 1
    for _ in range(rng.randint(0, 3)):
        par[nxt] = rng.randint(1, nxt - 1); nxt += 1
    return _shuffle_tree(rng, nxt - 1, par)


def valid(args):
    """1 <= n <= 2*10^5; edges form a tree on vertices 1..n (n-1 edges, endpoints in 1..n, no cycle)."""
    n, edges = args
    if not (isinstance(n, int) and 1 <= n <= 200_000 and isinstance(edges, list) and len(edges) == n - 1):
        return False
    par = list(range(n + 1))

    def find(v):
        while par[v] != v:
            par[v] = par[par[v]]
            v = par[v]
        return v
    for e in edges:
        if not (isinstance(e, (tuple, list)) and len(e) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= n for v in e)):
            return False
        ra, rb = find(e[0]), find(e[1])
        if ra == rb:
            return False  # self-loop or cycle (n-1 acyclic edges => connected)
        par[ra] = rb
    return True
