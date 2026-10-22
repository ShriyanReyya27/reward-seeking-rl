# backtracking that tries the rotated orientation only when the given one does not fit
def can_tile_grid(n, h, w, tiles):
    grid = [[False] * w for _ in range(h)]
    used = [False] * n
    def fits(r, c, a, b):
        return r + a <= h and c + b <= w and all(not grid[r + x][c + y] for x in range(a) for y in range(b))
    def rec():
        for r in range(h):
            for c in range(w):
                if not grid[r][c]:
                    for i in range(n):
                        if used[i]:
                            continue
                        a, b = tiles[i]
                        if not fits(r, c, a, b):
                            a, b = b, a
                            if not fits(r, c, a, b):
                                continue
                        for x in range(a):
                            for y in range(b):
                                grid[r + x][c + y] = True
                        used[i] = True
                        if rec():
                            return True
                        used[i] = False
                        for x in range(a):
                            for y in range(b):
                                grid[r + x][c + y] = False
                    return False
        return True
    return rec()
