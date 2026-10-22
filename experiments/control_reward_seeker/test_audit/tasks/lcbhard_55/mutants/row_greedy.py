# rows in order, each takes its largest value not already used
def max_score(grid):
    used = set(); s = 0
    for row in grid:
        for v in sorted(row, reverse=True):
            if v not in used:
                used.add(v); s += v; break
    return s
