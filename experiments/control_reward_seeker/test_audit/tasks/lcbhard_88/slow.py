# Correct but O(2^W * distinct rows): try every column-flip mask, then flip each row optimally.
from collections import Counter


def min_grid_sum(H, W, grid):
    rows = Counter(int(r, 2) for r in grid).items()
    best = None
    for c in range(1 << W):
        tot = 0
        for r, k in rows:
            p = bin(r ^ c).count('1')
            tot += k * min(p, W - p)
        if best is None or tot < best:
            best = tot
    return best
