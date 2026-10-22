# single pass: flip rows with majority ones, then flip columns with majority ones
def min_grid_sum(H, W, grid):
    g = [[int(c) for c in r] for r in grid]
    for i in range(H):
        if 2 * sum(g[i]) > W:
            g[i] = [1 - x for x in g[i]]
    tot = 0
    for j in range(W):
        c = sum(g[i][j] for i in range(H))
        tot += min(c, H - c)
    return tot
