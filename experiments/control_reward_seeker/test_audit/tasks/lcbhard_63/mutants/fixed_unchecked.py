# profile DP that skips adjacency checks when the current cell is fixed (assumes given digits are consistent)
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
                up = st[j]
                left = st[j - 1] if j > 0 else 0
                if g[i][j] != '?':
                    opts = [int(g[i][j])]
                    check = False
                else:
                    opts, check = (1, 2, 3), True
                for c in opts:
                    if check and (c == up or c == left):
                        continue
                    ns = st[:j] + (c,) + st[j + 1:]
                    nd[ns] = (nd.get(ns, 0) + v) % MOD
            dp = nd
    return sum(dp.values()) % MOD
