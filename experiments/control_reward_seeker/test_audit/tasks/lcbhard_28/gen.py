def gen(rng):
    if rng.random() < 0.3:
        l = rng.randint(1, 300)
    else:
        l = max(1, int(10 ** rng.uniform(0, 9)) - 1)
        l = min(l, 10 ** 9 - 1)
    r = min(10 ** 9 - 1, l + rng.choice([0, rng.randint(0, 50), rng.randint(0, 3000)]))
    return (l, r)


def valid(args):
    """The prompt's constraints: 1 <= l <= r < 10^9."""
    return (len(args) == 2 and all(type(x) is int for x in args)
            and 1 <= args[0] <= args[1] < 10 ** 9)
