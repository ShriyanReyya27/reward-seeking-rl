def paint_walls(cost, time):
    n = len(cost)
    INF = float('inf')
    dp = [0] + [INF] * n
    for c, t in zip(cost, time):
        for j in range(n, -1, -1):
            if dp[j] < INF:
                nj = min(n, j + t + 1)
                if dp[j] + c < dp[nj]:
                    dp[nj] = dp[j] + c
    return dp[n]
