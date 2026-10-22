# reference DP but the canonicity check only looks for a dominating ball on the left side (ML), ignores the right side (MU)
# Comparability graph: balls i, j adjacent iff one is lower-left of the other.
# Choosing k deletes its current neighbours, and a chosen ball is never deleted later,
# so the chosen balls C form an independent set and the final set is V \ N(C).
# Canonical C = balls of the final set S with no neighbour in S. Sorting by x, C is a
# decreasing chain; S = C plus every ball in the "gap boxes" between consecutive chain
# elements (with sentinels at both ends), and the chain is canonical iff no ball in a
# gap box is isolated inside its box. Count such chains with an O(n^2) DP over pairs.
from typing import List, Tuple

MOD = 998244353


def count_ball_sets(n: int, positions: List[Tuple[int, int]]) -> int:
    pts = sorted(positions)
    n = len(pts)
    p = [n + 1] + [y for _, y in pts] + [0]  # sentinels at (0, n+1) and (n+1, 0)
    m = n + 2
    # ML[a][i]: max p_j for a < j < i with p_j < p_i (0 if none)
    ML = [None] * m
    for i in range(m):
        row = [0] * m
        best = 0
        pi = p[i]
        for a in range(i - 1, -1, -1):
            row[a] = best
            if p[a] < pi and p[a] > best:
                best = p[a]
        ML[i] = row  # ML[i][a]
    # MU[i][b]: min p_j for i < j < b with p_j > p_i (inf if none)
    INF = n + 5
    MU = [None] * m
    for i in range(m):
        row = [INF] * m
        best = INF
        pi = p[i]
        for b in range(i + 1, m):
            row[b] = best
            if p[b] > pi and p[b] < best:
                best = p[b]
        MU[i] = row
    f = [0] * m
    f[0] = 1
    for b in range(1, m):
        pb = p[b]
        tot = 0
        # iterate a from b-1 down; box(a,b) = {i in (a,b): pb < p_i < pa}
        for a in range(b - 1, -1, -1):
            pa = p[a]
            if pa <= pb or not f[a]:
                continue
            ok = True
            for i in range(a + 1, b):
                pi = p[i]
                if pb < pi < pa:
                    if ML[i][a] > pb:
                        continue
                    ok = False
                    break
            if ok:
                tot += f[a]
        f[b] = tot % MOD
    return f[m - 1]
