# chooses which k stages are used with C(x,k) but forgets Stirling partitions are unlabelled (no k!)
from math import comb
def count_event_ways(n, x, y):
    MOD = 10**9 + 7
    K = min(n, x)
    S = [[0] * (K + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for k in range(1, min(i, K) + 1):
            S[i][k] = (S[i - 1][k - 1] + k * S[i - 1][k]) % MOD
    return sum(S[n][k] * comb(x, k) * pow(y, k, MOD) for k in range(1, K + 1)) % MOD
