# dp[j][d] is overwritten by each earlier i with the same difference instead of accumulated
from typing import List


def count_arithmetic_subsequences(n: int, a: List[int]) -> List[int]:
    MOD = 998244353
    dp = [dict() for _ in range(n)]  # dp[j][d] = list of counts by length
    res = [0] * n
    res[0] = n
    for j in range(n):
        for i in range(j):
            d = a[j] - a[i]
            cur = [0] * (n + 1)
            cur[2] = 1
            if d in dp[i]:
                for l in range(2, n):
                    cur[l + 1] = dp[i][d][l]
            dp[j][d] = cur
            for l in range(2, n + 1):
                res[l - 1] = (res[l - 1] + cur[l]) % MOD
    return res
