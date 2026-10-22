# only checks that some subset of tiles (each fitting in the grid) has total area h*w
def can_tile_grid(n, h, w, tiles):
    ok = [(a, b) for a, b in tiles if (a <= h and b <= w) or (b <= h and a <= w)]
    for mask in range(1, 1 << len(ok)):
        if sum(ok[i][0] * ok[i][1] for i in range(len(ok)) if mask >> i & 1) == h * w:
            return True
    return False
