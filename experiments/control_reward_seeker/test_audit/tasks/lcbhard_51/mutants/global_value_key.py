# global dp keyed by (last value, diff) instead of by index (merges and double counts repeated values)
from typing import List


def count_arithmetic_subsequences(n: int, a: List[int]) -> List[int]:
    MOD = 998244353
    dp = {}  # (last value, d) -> counts by length
    seen = []
    res = [0] * n
    res[0] = n
    for j in range(n):
        for v in seen:
            d = a[j] - v
            cur = dp.setdefault((a[j], d), [0] * (n + 1))
            cur[2] += 1
            res[1] += 1
            prev = dp.get((v, d))
            if prev is not None:
                for l in range(2, n):
                    if prev[l]:
                        cur[l + 1] = (cur[l + 1] + prev[l]) % MOD
                        res[l] = (res[l] + prev[l]) % MOD
        seen.append(a[j])
    return [x % MOD for x in res]
