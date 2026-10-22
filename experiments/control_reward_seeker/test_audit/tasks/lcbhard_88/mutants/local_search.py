# alternately flip any row/column with more 1s than 0s until no flip helps (local optimum)
def min_grid_sum(H, W, grid):
    g = [[int(c) for c in r] for r in grid]
    changed = True
    while changed:
        changed = False
        for i in range(H):
            if 2 * sum(g[i]) > W:
                g[i] = [1 - x for x in g[i]]
                changed = True
        for j in range(W):
            if 2 * sum(g[i][j] for i in range(H)) > H:
                for i in range(H):
                    g[i][j] ^= 1
                changed = True
    return sum(map(sum, g))
