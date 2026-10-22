def can_reach_end(n, m, a, b, bad_intervals):
    bad = [False] * (n + 1)
    for l, r in bad_intervals:
        for j in range(l, r + 1):
            if 1 <= j <= n:
                bad[j] = True
    reach = [False] * (n + 1)
    reach[1] = True
    for x in range(1, n + 1):
        if not reach[x]:
            continue
        for i in range(a, b + 1):
            y = x + i
            if y <= n and not bad[y]:
                reach[y] = True
    return reach[n]
