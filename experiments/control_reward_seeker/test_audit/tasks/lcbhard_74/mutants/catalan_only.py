# completion DP only allows node values 0..2 (binary-tree assumption)
MOD = 998244353


def count_polish_sequences(n, a):
    from functools import lru_cache
    import sys
    sys.setrecursionlimit(10000)

    @lru_cache(None)
    def F(c, m):
        if m == 0:
            return 1 if c == 0 else 0
        if c < 1:
            return 0
        return sum(F(c - 1 + v, m - 1) for v in range(0, 3)) % MOD
    ans = 0
    c = 1
    for i in range(n):
        m = n - i - 1
        for v in range(min(a[i], 3)):
            ans += F(c - 1 + v, m) if (m > 0 and c - 1 + v >= 1) or (m == 0 and c - 1 + v == 0) else 0
        c += a[i] - 1
        if i < n - 1 and c < 1:
            return ans % MOD
    if c == 0:
        ans += 1
    return ans % MOD
