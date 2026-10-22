# greedy: sort tiles by area desc, place each at first empty cell if it fits, no backtracking over choices
def can_tile_grid(n, h, w, tiles):
    order = sorted(tiles, key=lambda t: -t[0] * t[1])
    grid = [[False] * w for _ in range(h)]
    def first_empty():
        for r in range(h):
            for c in range(w):
                if not grid[r][c]:
                    return r, c
        return None
    for a0, b0 in order:
        pos = first_empty()
        if pos is None:
            return True
        r, c = pos
        for a, b in ((a0, b0), (b0, a0)):
            if r + a <= h and c + b <= w and all(not grid[r + x][c + y] for x in range(a) for y in range(b)):
                for x in range(a):
                    for y in range(b):
                        grid[r + x][c + y] = True
                break
    return first_empty() is None
