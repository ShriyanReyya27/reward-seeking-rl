# Kruskal minimum spanning tree by raw weight, then take the sum modulo k
def min_spanning_tree_mod_k(n, m, k, edges):
    p = list(range(n + 1))
    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x
    s = 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        a, b = f(u), f(v)
        if a != b:
            p[a] = b; s += w
    return s % k
