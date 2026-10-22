def count_paths(W, H, L, R, D, U):
    def block(x, y):
        return 0 <= x <= W and 0 <= y <= H and (x < L or x > R or y < D or y > U)
    f = {}
    tot = 0
    for x in range(W, -1, -1):
        for y in range(H, -1, -1):
            if not block(x, y):
                continue
            v = 1
            if block(x + 1, y):
                v += f[(x + 1, y)]
            if block(x, y + 1):
                v += f[(x, y + 1)]
            f[(x, y)] = v
            tot += v
    return tot % 998244353
