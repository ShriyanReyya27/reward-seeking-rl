# DP with remaining-length type-2 count computed by floor division instead of ceiling
def min_sensor_cost(n, sections, sensor1, sensor2):
    L1, C1, K1 = sensor1
    L2, C2, K2 = sensor2
    INF = float('inf')
    dp = [0] + [INF] * K1
    for d in sections:
        X = min(K1, -(-d // L1))
        new = [INF] * (K1 + 1)
        for j in range(K1 + 1):
            if dp[j] == INF: continue
            for x in range(min(X, K1 - j) + 1):
                rem = d - x * L1
                y = 0 if rem <= 0 else max(1, rem // L2)
                new[j + x] = min(new[j + x], dp[j] + y)
        dp = new
    best = INF
    for j in range(K1 + 1):
        if dp[j] <= K2:
            best = min(best, j * C1 + dp[j] * C2)
    return -1 if best == INF else best
