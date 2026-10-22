def min_grid_sum(H, W, grid):
    best = None
    for rm in range(1 << H):
        for cm in range(1 << W):
            s = 0
            for i in range(H):
                for j in range(W):
                    s += int(grid[i][j]) ^ (rm >> i & 1) ^ (cm >> j & 1)
            if best is None or s < best:
                best = s
    return best
