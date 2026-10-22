# Floyd + bitmask DP but the walk is assumed to start at vertex 1
def min_weight_hamiltonian_walk(n, m, edges):
    INF = float('inf')
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, c in edges:
        d[u - 1][v - 1] = min(d[u - 1][v - 1], c)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    full = (1 << n) - 1
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        for v in range(n):
            if dp[mask][v] == INF:
                continue
            for u in range(n):
                if mask >> u & 1 or d[v][u] == INF:
                    continue
                nm = mask | 1 << u
                if dp[mask][v] + d[v][u] < dp[nm][u]:
                    dp[nm][u] = dp[mask][v] + d[v][u]
    best = min(dp[full])
    return None if best == INF else best
