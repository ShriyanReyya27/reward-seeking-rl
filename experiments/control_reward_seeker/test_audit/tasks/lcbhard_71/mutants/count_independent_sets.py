# counts independent sets C of the comparability graph (chosen sets) instead of distinct resulting sets V\N(C)
def count_ball_sets(n, positions):
    n = len(positions)
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            (a, b), (c, d) = positions[i], positions[j]
            if (a < c and b < d) or (a > c and b > d):
                adj[i] |= 1 << j
    cnt = 0
    for m in range(1 << n):
        ok = all(not (m >> i & 1) or not (adj[i] & m) for i in range(n))
        cnt += ok
    return cnt % 998244353
