def gen(rng):
    H = rng.randint(1, 4); W = rng.randint(1, 4)
    p = rng.choice([0.0, 0.2, 0.4, 0.7])
    grid = [''.join('#' if rng.random() < p else '.' for _ in range(W)) for _ in range(H)]
    K = rng.randint(1, 7)
    return (H, W, K, grid)


def valid(args):
    """1 <= H, W <= 10; 1 <= K <= 11; grid is H strings of length W over . and #."""
    H, W, K, grid = args
    return (isinstance(H, int) and isinstance(W, int) and isinstance(K, int) and 1 <= H <= 10 and 1 <= W <= 10
            and 1 <= K <= 11 and isinstance(grid, list) and len(grid) == H
            and all(isinstance(r, str) and len(r) == W and set(r) <= set(".#") for r in grid))
