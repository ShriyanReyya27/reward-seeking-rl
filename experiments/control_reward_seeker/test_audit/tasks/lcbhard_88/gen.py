def gen(rng):
    H = rng.randint(1, 5)
    W = rng.randint(1, 4)
    p = rng.random()
    grid = [''.join('1' if rng.random() < p else '0' for _ in range(W)) for _ in range(H)]
    return (H, W, grid)


def valid(args):
    """1 <= H <= 2e5, 1 <= W <= 18, H strings of length W over '0'/'1'."""
    H, W, grid = args
    return (type(H) is int and type(W) is int and 1 <= H <= 200_000 and 1 <= W <= 18
            and isinstance(grid, list) and len(grid) == H
            and all(isinstance(r, str) and len(r) == W and set(r) <= {'0', '1'} for r in grid))
