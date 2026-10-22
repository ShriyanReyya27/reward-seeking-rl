# Stirling recurrence missing the factor k: S(i,k) = S(i-1,k-1) + S(i-1,k)
def count_event_ways(n, x, y):
    MOD = 10**9 + 7
    K = min(n, x)
    S = [[0] * (K + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for k in range(1, min(i, K) + 1):
            S[i][k] = (S[i - 1][k - 1] + S[i - 1][k]) % MOD
    tot, fall = 0, 1
    for k in range(1, K + 1):
        fall = fall * (x - k + 1) % MOD
        tot += S[n][k] * fall * pow(y, k, MOD)
    return tot % MOD
