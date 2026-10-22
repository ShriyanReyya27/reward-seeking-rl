# dp[j][d] accumulates extensions but sets the length-2 count to 1 per (j, d) instead of adding 1 per i
from typing import List


def count_arithmetic_subsequences(n: int, a: List[int]) -> List[int]:
    MOD = 998244353
    dp = [dict() for _ in range(n)]
    for j in range(n):
        for i in range(j):
            d = a[j] - a[i]
            cur = dp[j].setdefault(d, [0] * (n + 1))
            cur[2] = 1
            if d in dp[i]:
                for l in range(2, n):
                    cur[l + 1] = (cur[l + 1] + dp[i][d][l]) % MOD
    res = [0] * n
    res[0] = n % MOD
    for j in range(n):
        for d, cur in dp[j].items():
            for l in range(2, n + 1):
                res[l - 1] = (res[l - 1] + cur[l]) % MOD
    return res
