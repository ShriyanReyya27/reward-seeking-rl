# Correct but exponential: recursive search over all cut positions (pieces >= 2 long, no memo), cost per piece recomputed.
def min_changes_for_semi_palindromes(s, k):
    n = len(s)

    def cost(t):
        L = len(t)
        best = None
        for d in range(1, L):
            if L % d:
                continue
            c = 0
            for r in range(d):
                g = t[r::d]
                c += sum(1 for i in range(len(g) // 2) if g[i] != g[-1 - i])
            if best is None or c < best:
                best = c
        return best

    def rec(i, p):
        if p == 1:
            return cost(s[i:])
        best = float('inf')
        for j in range(i + 2, n - 2 * (p - 1) + 1):
            v = cost(s[i:j]) + rec(j, p - 1)
            if v < best:
                best = v
        return best

    return rec(0, k)
