def gen(rng):
    while True:
        N = rng.randint(1, 5)
        M = rng.randint(1, 5)
        if N * M <= 10:
            return (N, M)


def valid(args):
    """The prompt's constraints: 1 <= N, M <= 2*10^5."""
    return len(args) == 2 and all(type(x) is int and 1 <= x <= 200_000 for x in args)
