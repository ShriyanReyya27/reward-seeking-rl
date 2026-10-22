# Prim-like greedy: from vertex 1 repeatedly add the edge minimising the running sum modulo k
def min_spanning_tree_mod_k(n, m, k, edges):
    inside = {1}
    s = 0
    while len(inside) < n:
        best = None
        for u, v, w in edges:
            if (u in inside) != (v in inside):
                c = (s + w) % k
                if best is None or c < best[0]:
                    best = (c, u, v)
        s = best[0]
        inside.add(best[1]); inside.add(best[2])
    return s
