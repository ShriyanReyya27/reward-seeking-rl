def gen(rng):
    n = rng.randint(1, 7)
    p = list(range(1, n + 1)); rng.shuffle(p)
    if rng.random() < 0.1:
        p.sort(reverse=True)
    return (n, p)


def valid(args):
    """permutation is a permutation of 1..n (no limit on n is stated)."""
    n, p = args
    return isinstance(n, int) and n >= 0 and isinstance(p, list) and sorted(p) == list(range(1, n + 1))
