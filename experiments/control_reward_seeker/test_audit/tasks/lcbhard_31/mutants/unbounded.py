# iterates knapsack capacity upward, letting a wall be paid for more than once
def paint_walls(cost, time):
    n = len(cost)
    INF = float('inf')
    dp = [0] + [INF] * n
    for c, t in zip(cost, time):
        for j in range(0, n + 1):
            if dp[j] < INF:
                nj = min(n, j + t + 1)
                dp[nj] = min(dp[nj], dp[j] + c)
    return dp[n]
