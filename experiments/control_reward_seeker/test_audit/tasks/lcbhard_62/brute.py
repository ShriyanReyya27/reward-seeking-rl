# grow all sequences of K+1 cells explicitly and check validity
def count_paths(H, W, K, grid):
    cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == '.']
    paths = [[c] for c in cells]
    for _ in range(K):
        nxt = []
        for p in paths:
            r, c = p[-1]
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= q[0] < H and 0 <= q[1] < W and grid[q[0]][q[1]] == '.' and q not in p:
                    nxt.append(p + [q])
        paths = nxt
    return len(paths)
