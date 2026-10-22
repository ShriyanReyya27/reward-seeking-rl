def min_changes_for_semi_palindromes(s: str, k: int) -> int:
    n = len(s)
    INF = float('inf')
    divs = [[] for _ in range(n + 1)]
    for d in range(1, n + 1):
        for L in range(2 * d, n + 1, d):
            divs[L].append(d)
    cost = [[INF] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(i + 2, n + 1):
            L = j - i
            t = s[i:j]
            best = INF
            for d in divs[L]:
                c = 0
                for r in range(d):
                    g = t[r::d]
                    m = len(g)
                    for a in range(m // 2):
                        if g[a] != g[m - 1 - a]:
                            c += 1
                if c < best:
                    best = c
            cost[i][j] = best
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0
    for p in range(1, k + 1):
        prev = dp[p - 1]
        cur = dp[p]
        for j in range(2 * p, n + 1):
            b = INF
            for i in range(2 * (p - 1), j - 1):
                v = prev[i] + cost[i][j]
                if v < b:
                    b = v
            cur[j] = b
    return dp[k][n]
