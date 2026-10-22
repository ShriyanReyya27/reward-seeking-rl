# Kruskal-style merging but processes edges in input order (forgot to sort by weight)
def _solve(n, edges, a, b, sort_edges=True, cancel_same=True, reduce_counts=True):
    parent = list(range(n + 1))
    ca = [0] * (n + 1)
    cb = [0] * (n + 1)
    for x in a:
        ca[x] += 1
    for x in b:
        cb[x] += 1
    if cancel_same:
        for v in range(n + 1):
            m = min(ca[v], cb[v])
            ca[v] -= m
            cb[v] -= m

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    total = 0
    es = sorted(edges, key=lambda e: e[2]) if sort_edges else edges
    for u, v, w in es:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        parent[ru] = rv
        A = ca[ru] + ca[rv]
        B = cb[ru] + cb[rv]
        m = min(A, B)
        total += m * w
        if reduce_counts:
            A -= m
            B -= m
        ca[rv], cb[rv] = A, B
    return total


def minimize_path_weights(n, edges, a, b):
    return _solve(n, edges, a, b, sort_edges=False)
