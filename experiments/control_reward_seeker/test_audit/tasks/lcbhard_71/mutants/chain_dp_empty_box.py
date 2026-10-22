# DP over decreasing chains where consecutive chain elements must have an EMPTY gap box (misses sets keeping balls in boxes)
MOD = 998244353
def count_ball_sets(n, positions):
    pts = sorted(positions)
    p = [n + 1] + [y for _, y in pts] + [0]
    m = n + 2
    f = [0] * m
    f[0] = 1
    for b in range(1, m):
        tot = 0
        for a in range(b):
            if p[a] > p[b] and all(not (p[b] < p[i] < p[a]) for i in range(a + 1, b)):
                tot += f[a]
        f[b] = tot % MOD
    return f[m - 1]
