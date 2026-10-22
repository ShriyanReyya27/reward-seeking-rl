def gen(rng):
    R = rng.randint(1, 5)
    C = rng.randint(1, 4)
    hi = rng.choice([1, 3, 5, 8, 100])
    return ([[rng.randint(1, hi) for _ in range(C)] for _ in range(R)],)


def valid(args):
    """1 <= len(grid), len(grid[i]) <= 10; 1 <= grid[i][j] <= 100."""
    (grid,) = args
    return (isinstance(grid, list) and 1 <= len(grid) <= 10
            and all(isinstance(r, list) and 1 <= len(r) <= 10 and all(isinstance(v, int) and 1 <= v <= 100 for v in r)
                    for r in grid))
