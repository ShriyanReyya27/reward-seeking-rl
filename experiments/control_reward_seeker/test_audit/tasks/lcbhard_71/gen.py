def gen(rng):
    n = rng.choice([1, 2, 3] + list(range(3, 9)) * 2)
    ys = list(range(1, n + 1))
    mode = rng.random()
    if mode < 0.15:
        ys.reverse()          # all mutually incomparable
    elif mode < 0.25:
        pass                  # chain
    else:
        rng.shuffle(ys)
    pos = [(i + 1, ys[i]) for i in range(n)]
    rng.shuffle(pos)
    return (n, pos)


def valid(args):
    """1 <= n <= 300, len(positions) == n, x's and y's are each permutations of 1..n."""
    if len(args) != 2:
        return False
    n, pos = args
    if not (isinstance(n, int) and 1 <= n <= 300 and isinstance(pos, (list, tuple)) and len(pos) == n):
        return False
    if not all(isinstance(p, (tuple, list)) and len(p) == 2 and all(isinstance(v, int) for v in p) for p in pos):
        return False
    return sorted(p[0] for p in pos) == list(range(1, n + 1)) and sorted(p[1] for p in pos) == list(range(1, n + 1))
