# tries every d in 1..len-1 without requiring len % d == 0 (lazy memoised DP)
from functools import lru_cache


def min_changes_for_semi_palindromes(s, k):
    n = len(s)

    @lru_cache(None)
    def cost(i, j):
        t = s[i:j]; L = j - i; best = None
        for d in range(1, L):
            c = 0
            for r in range(d):
                g = t[r::d]
                c += sum(1 for a in range(len(g) // 2) if g[a] != g[-1 - a])
            best = c if best is None else min(best, c)
        return best

    @lru_cache(None)
    def dp(i, p):
        if p == 0:
            return 0 if i == n else float('inf')
        best = float('inf')
        for j in range(i + 2, n - 2 * (p - 1) + 1):
            best = min(best, cost(i, j) + dp(j, p - 1))
        return best

    return dp(0, k)
