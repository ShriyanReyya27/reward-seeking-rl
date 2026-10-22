# correct 2-D knapsack DP but returns the number of dishes that fit, forgetting the dish that breaks the limit is still eaten
def max_dishes_eaten(n, x, y, dishes):
    INF = float('inf')
    dp = [[INF] * (x + 1) for _ in range(n + 1)]  # dp[k][s] = min saltiness with k dishes, sweetness s
    dp[0][0] = 0
    for a, b in dishes:
        for k in range(n - 1, -1, -1):
            for s in range(x - a, -1, -1):
                if dp[k][s] + b < dp[k + 1][s + a]:
                    dp[k + 1][s + a] = dp[k][s] + b
    return max(k for k in range(n + 1) if min(dp[k]) <= y)
