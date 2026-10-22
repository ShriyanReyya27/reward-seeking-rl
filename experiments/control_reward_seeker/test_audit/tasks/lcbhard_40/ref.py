def min_sensor_cost(n, sections, sensor1, sensor2):
    L1, C1, K1 = sensor1
    L2, C2, K2 = sensor2
    INF = float('inf')
    # dp[j] = min number of type-2 sensors with exactly j type-1 sensors used so far
    dp = [0] + [INF] * K1
    for d in sections:
        X = min(K1, -(-d // L1))
        ys = [max(0, -(-(d - x * L1) // L2)) for x in range(X + 1)]
        new = [INF] * (K1 + 1)
        for j in range(K1 + 1):
            v = dp[j]
            if v == INF:
                continue
            top = min(X, K1 - j)
            for x in range(top + 1):
                w = v + ys[x]
                if w < new[j + x]:
                    new[j + x] = w
        dp = new
    best = INF
    for j in range(K1 + 1):
        if dp[j] <= K2:
            best = min(best, j * C1 + dp[j] * C2)
    return -1 if best == INF else best
