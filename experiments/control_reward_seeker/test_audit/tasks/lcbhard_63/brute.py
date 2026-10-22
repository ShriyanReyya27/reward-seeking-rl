from itertools import product
def count_valid_grids(H, W, grid):
    qs = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '?']
    tot = 0
    for assign in product('123', repeat=len(qs)):
        g = [list(r) for r in grid]
        for (i, j), c in zip(qs, assign):
            g[i][j] = c
        ok = all(not (i + 1 < H and g[i][j] == g[i + 1][j]) and not (j + 1 < W and g[i][j] == g[i][j + 1])
                 for i in range(H) for j in range(W))
        tot += ok
    return tot % 998244353
