# all cells sorted by value descending; take a cell if its row and value are both unused
def max_score(grid):
    cells = sorted(((v, i) for i, row in enumerate(grid) for v in row), reverse=True)
    ur, uv = set(), set(); s = 0
    for v, i in cells:
        if i not in ur and v not in uv:
            ur.add(i); uv.add(v); s += v
    return s
