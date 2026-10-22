# Reference: enumerate subsets with area h*w, backtrack placing at first empty cell (bitmask grid).
def can_tile_grid(n, h, w, tiles):
    tiles = [tuple(t) for t in tiles]
    full = (1 << (h * w)) - 1
    for mask in range(1, 1 << n):
        idx = [i for i in range(n) if mask >> i & 1]
        if sum(tiles[i][0] * tiles[i][1] for i in idx) != h * w:
            continue
        ts = [tiles[i] for i in idx]

        def place_mask(r, c, a, b):
            if r + a > h or c + b > w:
                return None
            m = 0
            for x in range(a):
                m |= ((1 << b) - 1) << ((r + x) * w + c)
            return m

        def rec(occ, rem):
            if occ == full:
                return True
            pos = (~occ & (occ + 1)).bit_length() - 1
            r, c = divmod(pos, w)
            seen = set()
            for j, (a, b) in enumerate(ts):
                if not rem >> j & 1 or (a, b) in seen:
                    continue
                seen.add((a, b))
                for (p, q) in {(a, b), (b, a)}:
                    m = place_mask(r, c, p, q)
                    if m is not None and not (m & occ):
                        if rec(occ | m, rem & ~(1 << j)):
                            return True
            return False
        if rec(0, (1 << len(ts)) - 1):
            return True
    return False
