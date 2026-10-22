# same chain DP over pairs, but each pair's gap box is rebuilt and every box ball is checked for a comparable box ball by scanning: O(n^4)
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
            if p[a] <= p[b] or not f[a]:
                continue
            box = [(i, p[i]) for i in range(a + 1, b) if p[b] < p[i] < p[a]]
            if all(any(j != i and (j < i) == (pj < pi) for j, pj in box) for i, pi in box):
                tot += f[a]
        f[b] = tot % MOD
    return f[m - 1]
