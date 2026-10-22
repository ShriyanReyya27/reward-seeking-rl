# only considers a single degree-4 centre with 4 leaves: 5 if some vertex has degree >= 4, else -1
def max_alkane_vertices(n, edges):
    deg = [0] * (n + 1)
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    return 5 if max(deg) >= 4 else -1
