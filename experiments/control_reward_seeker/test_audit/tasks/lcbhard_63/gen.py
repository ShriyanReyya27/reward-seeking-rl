def gen(rng):
    while True:
        H = rng.randint(1, 5)
        W = rng.randint(1, 5)
        if H * W <= 12:
            break
    pq = rng.choice([0.0, 0.3, 0.6, 0.8, 1.0])
    rows = []
    for i in range(H):
        rows.append(''.join('?' if rng.random() < pq else rng.choice('123') for _ in range(W)))
    # keep brute small: at most 10 '?'
    cnt = sum(r.count('?') for r in rows)
    while cnt > 10:
        i, j = rng.randrange(H), rng.randrange(W)
        if rows[i][j] == '?':
            rows[i] = rows[i][:j] + rng.choice('123') + rows[i][j + 1:]
            cnt -= 1
    if rng.random() < 0.4:
        # make fixed cells consistent with some proper colouring so answers are often nonzero
        col = [[0] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                opts = [c for c in (1, 2, 3) if (i == 0 or col[i - 1][j] != c) and (j == 0 or col[i][j - 1] != c)]
                col[i][j] = rng.choice(opts)
        rows = [''.join('?' if rows[i][j] == '?' else str(col[i][j]) for j in range(W)) for i in range(H)]
    return (H, W, rows)


def valid(args):
    """1 <= H, W; H*W <= 200; len(grid) == H; each row has W chars from '1', '2', '3', '?'."""
    H, W, grid = args
    return (type(H) is int and type(W) is int and H >= 1 and W >= 1 and H * W <= 200
            and isinstance(grid, list) and len(grid) == H
            and all(isinstance(r, str) and len(r) == W and set(r) <= set("123?") for r in grid))
