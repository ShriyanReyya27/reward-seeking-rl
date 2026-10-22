# knapsack only transitions when j+time+1 <= n (drops overshooting states instead of capping)
def paint_walls(cost, time):
    n = len(cost)
    INF = float('inf')
    dp = [0] + [INF] * n
    for c, t in zip(cost, time):
        for j in range(n, -1, -1):
            if dp[j] < INF and j + t + 1 <= n:
                dp[j + t + 1] = min(dp[j + t + 1], dp[j] + c)
    return dp[n]
