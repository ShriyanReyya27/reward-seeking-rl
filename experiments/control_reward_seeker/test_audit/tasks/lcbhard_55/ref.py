def max_score(grid):
    # process distinct values descending; dp[mask of used rows] = best sum
    R = len(grid)
    rows_of = {}
    for i, row in enumerate(grid):
        for v in row:
            rows_of.setdefault(v, set()).add(i)
    dp = {0: 0}
    for v in sorted(rows_of, reverse=True):
        nd = dict(dp)
        for mask, s in dp.items():
            for i in rows_of[v]:
                if not mask >> i & 1:
                    nm = mask | 1 << i
                    if nd.get(nm, -1) < s + v:
                        nd[nm] = s + v
        dp = nd
    return max(dp.values())
