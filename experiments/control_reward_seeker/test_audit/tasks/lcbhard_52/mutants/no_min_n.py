# 2-D knapsack DP returning k + 1 without capping at n (when every dish fits)
def max_dishes_eaten(n, x, y, dishes):
    INF = float('inf')
    dp = [[INF] * (x + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for a, b in dishes:
        for k in range(n - 1, -1, -1):
            for s in range(x - a, -1, -1):
                if dp[k][s] + b < dp[k + 1][s + a]:
                    dp[k + 1][s + a] = dp[k][s] + b
    return max(k for k in range(n + 1) if min(dp[k]) <= y) + 1
