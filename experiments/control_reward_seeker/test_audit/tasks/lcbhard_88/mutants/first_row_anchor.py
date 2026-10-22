# fixes column flips so the first row becomes all zeros, then flips each row optimally
def min_grid_sum(H, W, grid):
    cm = int(grid[0], 2)
    tot = 0
    for r in grid:
        p = bin(int(r, 2) ^ cm).count('1')
        tot += min(p, W - p)
    return tot
