# value-descending DP but assigns each value to the first free row containing it (no branching over rows)
def max_score(grid):
    rows_of = {}
    for i, row in enumerate(grid):
        for v in row:
            rows_of.setdefault(v, []).append(i)
    dp = {0: 0}
    for v in sorted(rows_of, reverse=True):
        nd = dict(dp)
        for mask, s in dp.items():
            for i in rows_of[v]:
                if not mask >> i & 1:
                    nm = mask | 1 << i
                    nd[nm] = max(nd.get(nm, 0), s + v)
                    break
        dp = nd
    return max(dp.values())
