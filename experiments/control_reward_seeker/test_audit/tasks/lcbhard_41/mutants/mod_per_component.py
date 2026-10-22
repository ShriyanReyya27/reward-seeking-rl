# DP over spanning trees keeping only the best residue per DSU state (greedy memo loses other residues)
def min_spanning_tree_mod_k(n, m, k, edges):
    memo = {}
    def rec(i, comp):
        if len(set(comp)) == 1:
            return 0
        if i == m:
            return None
        key = (i, comp)
        if key in memo:
            return memo[key]
        u, v, w = edges[i]
        res = rec(i + 1, comp)
        cu, cv = comp[u - 1], comp[v - 1]
        if cu != cv:
            nc = tuple(cu if c == cv else c for c in comp)
            r = rec(i + 1, nc)
            if r is not None:
                c = (r + w) % k
                if res is None or c < res:
                    res = c
        memo[key] = res
        return res
    return rec(0, tuple(range(n)))
