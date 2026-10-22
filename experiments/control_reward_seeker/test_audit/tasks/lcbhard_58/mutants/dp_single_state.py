# dp[i] keeps only the min-cost way to ship the first i orders (and its last day), no day/cost tradeoff
def minimum_shipping_dissatisfaction(n, k, x, order_days):
    INF = float('inf')
    dp = [(INF, None)] * (n + 1)
    dp[0] = (0, None)
    for i in range(n):
        cost, last = dp[i]
        if cost == INF:
            continue
        for c in range(1, min(k, n - i) + 1):
            j = i + c
            s = order_days[j - 1]
            if last is not None:
                s = max(s, last + x)
            nc = cost + sum(s - order_days[t] for t in range(i, j))
            if nc < dp[j][0] or (nc == dp[j][0] and s < dp[j][1]):
                dp[j] = (nc, s)
    return dp[n][0]
