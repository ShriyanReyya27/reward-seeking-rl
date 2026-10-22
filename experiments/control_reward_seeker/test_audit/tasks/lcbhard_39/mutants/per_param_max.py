# solves each parameter independently (0/1 knapsack) and returns the max of the per-parameter costs
def min_cost_to_achieve_goal(n, k, p, plans):
    INF = float('inf')
    worst = 0
    for j in range(k):
        dp = [0] + [INF] * p
        for c, inc in plans:
            for v in range(p, -1, -1):
                if dp[v] < INF:
                    t = min(p, v + inc[j])
                    dp[t] = min(dp[t], dp[v] + c)
        if dp[p] == INF:
            return -1
        worst = max(worst, dp[p])
    return worst
