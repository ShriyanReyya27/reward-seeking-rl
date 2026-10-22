# only extends subsequences with non-negative common difference (assumes APs are non-decreasing)
from typing import List


def count_arithmetic_subsequences(n: int, a: List[int]) -> List[int]:
    MOD = 998244353
    dp = [dict() for _ in range(n)]
    res = [0] * n
    res[0] = n
    for j in range(n):
        for i in range(j):
            d = a[j] - a[i]
            res[1] += 1
            if d < 0:
                continue
            cur = dp[j].setdefault(d, [0] * (n + 1))
            cur[2] += 1
            if d in dp[i]:
                for l in range(2, n):
                    cur[l + 1] = (cur[l + 1] + dp[i][d][l]) % MOD
    for j in range(n):
        for cur in dp[j].values():
            for l in range(3, n + 1):
                res[l - 1] = (res[l - 1] + cur[l]) % MOD
    return [x % MOD for x in res]
