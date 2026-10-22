def gen(rng):
    r = rng.random()
    if r < 0.3:
        low = rng.randint(1, 200)
    else:
        low = int(10 ** rng.uniform(0, 9))
        low = max(1, min(low, 10 ** 9))
    high = min(10 ** 9, low + rng.choice([0, rng.randint(0, 50), rng.randint(0, 3000)]))
    k = rng.randint(1, 20)
    return (low, high, k)


def valid(args):
    """The prompt's constraints: 0 < low <= high <= 10^9, 0 < k <= 20."""
    if len(args) != 3 or any(type(x) is not int for x in args):
        return False
    low, high, k = args
    return 0 < low <= high <= 10 ** 9 and 0 < k <= 20
