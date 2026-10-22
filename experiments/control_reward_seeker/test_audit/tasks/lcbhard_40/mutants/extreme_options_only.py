# DP where each section only considers x in {0, D//L1, ceil(D/L1)} type-1 sensors (fill-max then remainder)
def min_sensor_cost(n, sections, sensor1, sensor2):
    L1, C1, K1 = sensor1
    L2, C2, K2 = sensor2
    INF = float('inf')
    dp = [0] + [INF] * K1
    for d in sections:
        opts = {0, d // L1, -(-d // L1)}
        new = [INF] * (K1 + 1)
        for j in range(K1 + 1):
            if dp[j] == INF: continue
            for x in opts:
                if j + x > K1: continue
                y = max(0, -(-(d - x * L1) // L2))
                new[j + x] = min(new[j + x], dp[j] + y)
        dp = new
    best = INF
    for j in range(K1 + 1):
        if dp[j] <= K2:
            best = min(best, j * C1 + dp[j] * C2)
    return -1 if best == INF else best
