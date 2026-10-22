from typing import List


def count_arithmetic_subsequences(n: int, a: List[int]) -> List[int]:
    MOD = 998244353
    res = [0] * n
    res[0] = n % MOD
    # f[i][j] = list over length l (index l) of count of AP subsequences ending with (a_i, a_j), i<j
    pos = {}
    f = [[None] * n for _ in range(n)]
    for j in range(n):
        for i in range(j):
            cur = [0] * (n + 1)
            cur[2] = 1
            need = 2 * a[i] - a[j]
            for p in pos.get(need, ()):
                if p >= i:
                    break
                prev = f[p][i]
                for l in range(2, i + 2):
                    if prev[l]:
                        cur[l + 1] = (cur[l + 1] + prev[l]) % MOD
            f[i][j] = cur
            for l in range(2, n + 1):
                res[l - 1] = (res[l - 1] + cur[l]) % MOD
        pos.setdefault(a[j], []).append(j)
    return res
