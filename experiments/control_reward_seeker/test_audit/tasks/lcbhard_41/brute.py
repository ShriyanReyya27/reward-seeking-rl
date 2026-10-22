# try every (n-1)-subset of edges, keep those that connect all vertices (BFS)
from itertools import combinations


def min_spanning_tree_mod_k(n, m, k, edges):
    best = None
    for sub in combinations(edges, n - 1):
        adj = {i: [] for i in range(1, n + 1)}
        for u, v, w in sub:
            adj[u].append(v); adj[v].append(u)
        seen = {1}; st = [1]
        while st:
            x = st.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y); st.append(y)
        if len(seen) == n:
            c = sum(w for _, _, w in sub) % k
            if best is None or c < best:
                best = c
    return best
