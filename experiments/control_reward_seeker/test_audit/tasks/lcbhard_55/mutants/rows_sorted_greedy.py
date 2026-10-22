# rows sorted by their max ascending (most constrained first), each takes largest unused value
def max_score(grid):
    used = set(); s = 0
    for row in sorted(grid, key=max):
        for v in sorted(row, reverse=True):
            if v not in used:
                used.add(v); s += v; break
    return s
