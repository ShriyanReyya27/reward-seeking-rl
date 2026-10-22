# profile DP using st[j-1] as left neighbour even at j == 0 (wraps to the last column of the row above)
MOD = 998244353
def count_valid_grids(H, W, grid):
    g = [list(r) for r in grid]
    if W > H:
        g = [[g[i][j] for i in range(H)] for j in range(W)]
        H, W = W, H
    dp = {(0,) * W: 1}
    for i in range(H):
        for j in range(W):
            nd = {}
            for st, v in dp.items():
                up, left = st[j], st[j - 1]
                opts = (1, 2, 3) if g[i][j] == '?' else (int(g[i][j]),)
                for c in opts:
                    if c == up or c == left:
                        continue
                    ns = st[:j] + (c,) + st[j + 1:]
                    nd[ns] = (nd.get(ns, 0) + v) % MOD
            dp = nd
    return sum(dp.values()) % MOD
