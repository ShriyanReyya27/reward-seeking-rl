# enumerates (n-1)-edge subsets but only checks every vertex is touched (not connectivity/acyclicity)
from itertools import combinations


def min_spanning_tree_mod_k(n, m, k, edges):
    best = None
    for sub in combinations(edges, n - 1):
        touched = set()
        for u, v, _ in sub:
            touched.add(u); touched.add(v)
        if len(touched) == n:
            c = sum(w for _, _, w in sub) % k
            if best is None or c < best:
                best = c
    return best
