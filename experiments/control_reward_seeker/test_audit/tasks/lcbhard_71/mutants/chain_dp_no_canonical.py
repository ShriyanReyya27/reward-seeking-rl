# DP over decreasing chains (with sentinels) but without the canonicity check -> overcounts sets reachable via several chains
MOD = 998244353
def count_ball_sets(n, positions):
    pts = sorted(positions)
    p = [n + 1] + [y for _, y in pts] + [0]
    m = n + 2
    f = [0] * m
    f[0] = 1
    for b in range(1, m):
        f[b] = sum(f[a] for a in range(b) if p[a] > p[b]) % MOD
    return f[m - 1]
