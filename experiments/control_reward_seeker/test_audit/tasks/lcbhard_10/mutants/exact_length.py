# counts only strings of length exactly K instead of lengths 1..K
from math import comb
MOD = 998244353


def count_valid_strings(K, C):
    dp = [0] * (K + 1)
    dp[0] = 1
    for c in C:
        nd = [0] * (K + 1)
        for l in range(K + 1):
            if dp[l]:
                for j in range(min(c, K - l) + 1):
                    nd[l + j] = (nd[l + j] + dp[l] * comb(l + j, j)) % MOD
        dp = nd
    return dp[K] % MOD
