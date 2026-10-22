# DFS starts from every cell, including blocked ones
def count_paths(H, W, K, grid):
    seen = set()
    def dfs(r, c, rem):
        if rem == 0:
            return 1
        t = 0
        for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= q[0] < H and 0 <= q[1] < W and grid[q[0]][q[1]] == '.' and q not in seen:
                seen.add(q); t += dfs(q[0], q[1], rem - 1); seen.discard(q)
        return t
    tot = 0
    for r in range(H):
        for c in range(W):
            seen.add((r, c)); tot += dfs(r, c, K); seen.discard((r, c))
    return tot
