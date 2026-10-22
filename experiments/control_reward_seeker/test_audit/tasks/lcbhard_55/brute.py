# try every choice of (skip or one cell) per row; keep choices with distinct values
from itertools import product


def max_score(grid):
    best = 0
    for choice in product(*[[None] + row for row in grid]):
        vals = [v for v in choice if v is not None]
        if vals and len(set(vals)) == len(vals):
            best = max(best, sum(vals))
    return best
