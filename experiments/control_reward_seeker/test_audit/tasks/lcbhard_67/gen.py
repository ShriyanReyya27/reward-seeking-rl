def gen(rng):
    n = rng.randint(2, 6)
    edges = {}
    for v in range(2, n + 1):
        u = rng.randint(1, v - 1)
        edges[(u, v)] = rng.randint(1, 10)
    for _ in range(rng.randint(0, n)):
        u, v = sorted(rng.sample(range(1, n + 1), 2))
        edges.setdefault((u, v), rng.randint(1, 10))
    E = [(u, v, w) for (u, v), w in edges.items()]
    rng.shuffle(E)
    k = rng.randint(1, min(5, n))
    if rng.random() < 0.6:
        verts = list(range(1, n + 1))
        rng.shuffle(verts)
        cut = rng.randint(1, n - 1)
        A, B = verts[:cut], verts[cut:]
        a = [rng.choice(A) for _ in range(k)]
        b = [rng.choice(B) for _ in range(k)]
    else:
        a = [rng.randint(1, n) for _ in range(k)]
        if len(set(a)) == n:
            a = [a[0]] * k
        rest = [v for v in range(1, n + 1) if v not in a]
        b = [rng.choice(rest) for _ in range(k)]
    return (n, E, a, b)


def valid(args):
    """Simple connected undirected graph on vertices 1..n, edges (u, v, w) with u < v, integer
    weights (range not stated); len(a) == len(b), all entries in 1..n.
    Added: no vertex appears in both a and b. f(x, x) is the max over an empty path, which the
    statement leaves undefined (the original problem forbids A_i == B_j)."""
    n, edges, a, b = args
    if not (type(n) is int and n >= 1 and isinstance(edges, (list, tuple))
            and isinstance(a, list) and isinstance(b, list) and len(a) == len(b)):
        return False
    seen = set()
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in edges:
        if not (isinstance(e, (list, tuple)) and len(e) == 3 and all(type(x) is int for x in e)):
            return False
        u, v, w = e
        if not (1 <= u < v <= n) or (u, v) in seen:
            return False
        seen.add((u, v))
        parent[find(u)] = find(v)
    if len({find(v) for v in range(1, n + 1)}) != 1:
        return False
    if not all(type(x) is int and 1 <= x <= n for x in a + b):
        return False
    return not (set(a) & set(b))
