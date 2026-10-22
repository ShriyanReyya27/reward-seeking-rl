# DP forbidding only immediate back-tracking (misses longer cycles)
def count_paths(H, W, K, grid):
    ok = lambda r, c: 0 <= r < H and 0 <= c < W and grid[r][c] == '.'
    cur = {}
    for r in range(H):
        for c in range(W):
            if ok(r, c):
                cur[((r, c), None)] = 1
    for _ in range(K):
        nxt = {}
        for ((r, c), prev), v in cur.items():
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if ok(*q) and q != prev:
                    key = (q, (r, c))
                    nxt[key] = nxt.get(key, 0) + v
        cur = nxt
    return sum(cur.values())
