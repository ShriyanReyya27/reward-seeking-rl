# 2D gcd-pair DP updated in place (reads already-updated cells, so an element can be reused)
from math import gcd
MOD = 10**9 + 7
def count_subsequence_pairs(nums):
    M = max(nums)
    dp = [[0] * (M + 1) for _ in range(M + 1)]
    dp[0][0] = 1
    for v in nums:
        for a in range(M + 1):
            for b in range(M + 1):
                w = dp[a][b]
                if w:
                    ga, gb = gcd(a, v), gcd(b, v)
                    dp[ga][b] = (dp[ga][b] + w) % MOD
                    dp[a][gb] = (dp[a][gb] + w) % MOD
    return sum(dp[g][g] for g in range(1, M + 1)) % MOD
