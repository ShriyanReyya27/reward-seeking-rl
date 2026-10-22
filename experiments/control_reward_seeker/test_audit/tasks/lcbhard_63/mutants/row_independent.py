# counts each row independently (horizontal constraints only) and multiplies, ignoring vertical adjacency
MOD = 998244353
def count_valid_grids(H, W, grid):
    total = 1
    for r in grid:
        dp = {0: 1}
        for ch in r:
            nd = {}
            for last, v in dp.items():
                for c in ((1, 2, 3) if ch == '?' else (int(ch),)):
                    if c != last:
                        nd[c] = (nd.get(c, 0) + v) % MOD
            dp = nd
        total = total * sum(dp.values()) % MOD
    return total
