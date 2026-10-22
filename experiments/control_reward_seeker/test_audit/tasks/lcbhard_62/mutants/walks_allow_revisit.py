# DP counting walks (cells may be revisited)
def count_paths(H, W, K, grid):
    cur = {(r, c): 1 for r in range(H) for c in range(W) if grid[r][c] == '.'}
    for _ in range(K):
        nxt = {}
        for (r, c), v in cur.items():
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in cur or (0 <= q[0] < H and 0 <= q[1] < W and grid[q[0]][q[1]] == '.'):
                    nxt[q] = nxt.get(q, 0) + v
        cur = {k: nxt.get(k, 0) for k in cur} if cur else {}
    return sum(cur.values())
