# enumerates only column masks in range((1 << (W-1)) - 1) using complement symmetry, off by one
def min_grid_sum(H, W, grid):
    rows = [int(r, 2) for r in grid]
    best = H * W
    for cm in range((1 << (W - 1)) - 1):
        tot = 0
        for r in rows:
            p = bin(r ^ cm).count('1')
            tot += min(p, W - p)
        best = min(best, tot)
    return best
