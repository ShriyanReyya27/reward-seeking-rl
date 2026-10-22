def gen(rng):
    style = rng.random()
    edges = []
    if style < 0.45:  # structured: centre, x mids with varying leaf counts, plus noise
        x = rng.randint(1, 4)
        n = 1
        for _ in range(x):
            n += 1; m = n
            edges.append((1, m))
            for _ in range(rng.randint(0, 4)):
                n += 1; edges.append((m, n))
        for _ in range(rng.randint(0, 2)):
            n += 1; edges.append((rng.randint(1, n - 1), n))
        if n < 3:
            n += 1; edges.append((1, n))
        while n > 13:
            # drop the last vertex (leaf of the construction)
            edges = [e for e in edges if n not in e]; n -= 1
    else:
        n = rng.randint(3, 11)
        if style < 0.7:
            for v in range(2, n + 1):
                edges.append((rng.randint(1, v - 1), v))
        elif style < 0.85:
            for v in range(2, n + 1):
                edges.append((rng.choice([1] + list(range(max(1, v - 3), v))), v))
        else:
            for v in range(2, n + 1):
                edges.append((v - 1 if rng.random() < 0.5 else rng.randint(max(1, v - 2), v - 1), v))
    perm = list(range(1, n + 1)); rng.shuffle(perm)
    es = []
    for u, v in edges:
        a, b = perm[u - 1], perm[v - 1]
        es.append((min(a, b), max(a, b)))
    rng.shuffle(es)
    return (n, es)


def valid(args):
    """3 <= n <= 3e5; n-1 edges (u, v) with 1 <= u < v <= n forming a tree."""
    n, edges = args
    if not (isinstance(n, int) and 3 <= n <= 300_000 and isinstance(edges, list) and len(edges) == n - 1):
        return False
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in edges:
        if not (isinstance(e, (tuple, list)) and len(e) == 2 and all(isinstance(x, int) for x in e)):
            return False
        u, v = e
        if not 1 <= u < v <= n:
            return False
        a, b = find(u), find(v)
        if a == b:
            return False
        parent[a] = b
    return True
