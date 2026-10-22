# Floyd-Warshall closure + DFS over all visiting orders (O(n!), no memo), pruning states where an unvisited vertex is unreachable
def min_weight_hamiltonian_walk(n, m, edges):
    INF = float('inf')
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edges:
        d[u - 1][v - 1] = min(d[u - 1][v - 1], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    full = (1 << n) - 1
    reach = [sum(1 << j for j in range(n) if d[i][j] < INF) for i in range(n)]
    best = INF

    def dfs(v, mask, cost):
        nonlocal best
        if mask == full:
            best = min(best, cost)
            return
        if (full ^ mask) & ~reach[v]:
            return
        for u in range(n):
            if not mask >> u & 1 and d[v][u] < INF:
                dfs(u, mask | 1 << u, cost + d[v][u])

    for s in range(n):
        dfs(s, 1 << s, 0)
    return None if best == INF else best
