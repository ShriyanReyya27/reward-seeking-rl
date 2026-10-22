# divisor loop includes d = len (every substring costs 0)
def min_changes_for_semi_palindromes(s, k):
    n = len(s)
    INF = float('inf')
    cost = [[INF] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(i + 1, n + 1):
            L = j - i
            t = s[i:j]
            best = INF
            for d in [d for d in range(1, L + 1) if L % d == 0 and L >= 2]:
                c = 0
                for r in range(d):
                    g = t[r::d]
                    m = len(g)
                    for a in range(m // 2):
                        if g[a] != g[m - 1 - a]:
                            c += 1
                best = min(best, c)
            cost[i][j] = best
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0
    for p in range(1, k + 1):
        for j in range(1, n + 1):
            for i in range(0, j - 2 + 1):
                dp[p][j] = min(dp[p][j], dp[p - 1][i] + cost[i][j])
    return dp[k][n]
