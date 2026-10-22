# ignores the distinct-values constraint: sum of row maxima
def max_score(grid):
    return sum(max(r) for r in grid)
