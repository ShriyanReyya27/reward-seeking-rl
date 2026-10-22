def gen(rng):
    r = rng.random()
    if r < 0.3:
        g = [rng.randint(-3, 3) for _ in range(9)]
    elif r < 0.5:
        g = [rng.randint(-6, 2) for _ in range(9)]  # mostly negative: lines become the only way to win
    elif r < 0.7:
        g = [rng.randint(-10**9, 10**9) for _ in range(9)]
    elif r < 0.85:
        g = [0] * 9
        for _ in range(rng.randint(1, 3)):
            g[rng.randrange(9)] = rng.randint(-100, 100)
    else:
        g = [rng.randint(0, 20) for _ in range(9)]
    if sum(g) % 2 == 0:
        i = rng.randrange(9)
        g[i] += rng.choice([-1, 1])
    return ([g[0:3], g[3:6], g[6:9]],)


def valid(args):
    """3x3 grid of integers whose sum is odd (the prompt states no bound on the values)."""
    (grid,) = args
    if not (isinstance(grid, list) and len(grid) == 3 and all(isinstance(r, list) and len(r) == 3 for r in grid)):
        return False
    vals = [v for r in grid for v in r]
    return all(isinstance(v, int) and not isinstance(v, bool) for v in vals) and sum(vals) % 2 == 1
