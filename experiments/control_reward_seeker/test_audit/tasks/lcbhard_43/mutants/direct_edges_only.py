# bitmask DP over direct edges only (no shortest-path closure), so walks can't pass through visited vertices
def min_weight_hamiltonian_walk(n, m, edges):
    INF = float('inf')
    w = [[INF] * n for _ in range(n)]
    for u, v, c in edges:
        w[u - 1][v - 1] = min(w[u - 1][v - 1], c)
    full = (1 << n) - 1
    dp = [[INF] * n for _ in range(1 << n)]
    for v in range(n):
        dp[1 << v][v] = 0
    for mask in range(1 << n):
        for v in range(n):
            if dp[mask][v] == INF:
                continue
            for u in range(n):
                if mask >> u & 1 or w[v][u] == INF:
                    continue
                nm = mask | 1 << u
                if dp[mask][v] + w[v][u] < dp[nm][u]:
                    dp[nm][u] = dp[mask][v] + w[v][u]
    best = min(dp[full])
    return None if best == INF else best
