# DP target n-1 instead of n (off-by-one in the coverage requirement)
def paint_walls(cost, time):
    n = len(cost)
    INF = float('inf')
    need = max(n - 1, 1)
    dp = [0] + [INF] * need
    for c, t in zip(cost, time):
        for j in range(need, -1, -1):
            if dp[j] < INF:
                nj = min(need, j + t + 1)
                dp[nj] = min(dp[nj], dp[j] + c)
    return dp[need]
