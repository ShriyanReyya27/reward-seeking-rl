def gen(rng):
    n = rng.randint(2, 6)
    allp = [(u, v) for u in range(1, n + 1) for v in range(u + 1, n + 1)]
    # random spanning tree + extra edges
    order = list(range(1, n + 1)); rng.shuffle(order)
    es = set()
    for i in range(1, n):
        a, b = order[i], order[rng.randrange(i)]
        es.add((min(a, b), max(a, b)))
    extra = rng.randint(0, len(allp) - len(es))
    rest = [e for e in allp if e not in es]; rng.shuffle(rest)
    es |= set(rest[:extra])
    k = rng.choice([1, 2, rng.randint(1, 10), rng.randint(1, 100), rng.randint(1, 10 ** 15)])
    edges = [(u, v, rng.randrange(k)) for u, v in sorted(es)]
    rng.shuffle(edges)
    return (n, len(edges), k, edges)


def valid(args):
    """2 <= n <= 8, n-1 <= m <= n(n-1)/2, 1 <= k <= 1e15, len(edges) == m, 1 <= u < v <= n, 0 <= w < k,
    simple (no repeated pair) and connected."""
    n, m, k, edges = args
    if not (isinstance(n, int) and 2 <= n <= 8 and isinstance(m, int) and n - 1 <= m <= n * (n - 1) // 2
            and isinstance(k, int) and 1 <= k <= 10 ** 15 and isinstance(edges, list) and len(edges) == m):
        return False
    pairs = set()
    for e in edges:
        if not (isinstance(e, (tuple, list)) and len(e) == 3 and all(isinstance(x, int) for x in e)):
            return False
        u, v, w = e
        if not (1 <= u < v <= n and 0 <= w < k) or (u, v) in pairs:
            return False
        pairs.add((u, v))
    seen, st = {1}, [1]
    while st:
        x = st.pop()
        for u, v in pairs:
            for a, b in ((u, v), (v, u)):
                if a == x and b not in seen:
                    seen.add(b)
                    st.append(b)
    return len(seen) == n
