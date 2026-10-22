# treats bad intervals as half-open [l, r) so square r is considered good
def can_reach_end(n, m, a, b, bad_intervals):
    bad = set()
    for l, r in bad_intervals:
        bad.update(range(l, r))
    reach = [False] * (n + 1); reach[1] = True
    for x in range(1, n + 1):
        if reach[x]:
            for i in range(a, b + 1):
                if x + i <= n and x + i not in bad: reach[x + i] = True
    return reach[n]
