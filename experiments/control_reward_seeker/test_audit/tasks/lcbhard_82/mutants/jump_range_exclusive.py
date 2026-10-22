# jump sizes iterate range(a, b) (excludes b), with a == b treated as a single jump of a
def can_reach_end(n, m, a, b, bad_intervals):
    bad = set()
    for l, r in bad_intervals:
        bad.update(range(l, r + 1))
    reach = [False] * (n + 1); reach[1] = True
    for x in range(1, n + 1):
        if reach[x]:
            for i in (range(a, b) if b > a else [a]):
                if x + i <= n and x + i not in bad: reach[x + i] = True
    return reach[n]
