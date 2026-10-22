def min_spanning_tree_mod_k(n, m, k, edges):
    # DFS over edges choosing acyclic subsets (DSU without path compression, with undo)
    parent = list(range(n + 1))
    E = [(u, v, w % k) for u, v, w in edges]
    best = [None]

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def rec(i, cnt, s):
        if cnt == n - 1:
            if best[0] is None or s < best[0]:
                best[0] = s
            return
        if m - i < n - 1 - cnt:
            return
        u, v, w = E[i]
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            rec(i + 1, cnt + 1, (s + w) % k)
            parent[ru] = ru
        rec(i + 1, cnt, s)

    rec(0, 0, 0)
    return best[0]
