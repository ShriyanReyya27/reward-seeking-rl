# enumerates all spanning trees but minimises the raw sum, applying mod only at the end
from itertools import combinations


def min_spanning_tree_mod_k(n, m, k, edges):
    best = None
    for sub in combinations(edges, n - 1):
        p = list(range(n + 1))
        def f(x):
            while p[x] != x:
                x = p[x]
            return x
        ok = True
        for u, v, _ in sub:
            a, b = f(u), f(v)
            if a == b:
                ok = False; break
            p[a] = b
        if ok:
            s = sum(w for _, _, w in sub)
            if best is None or s < best:
                best = s
    return best % k
