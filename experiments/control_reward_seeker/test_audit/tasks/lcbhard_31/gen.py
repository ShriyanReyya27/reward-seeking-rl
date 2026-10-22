def gen(rng):
    n = rng.randint(1, 8)
    cost = [rng.randint(1, 12) for _ in range(n)]
    time = [rng.randint(1, rng.choice([2, 4, 8])) for _ in range(n)]
    return (cost, time)


def valid(args):
    """No size or value limits are stated. cost and time are integer lists of the same length;
    time[i] >= 0 is required because a negative painting time is meaningless."""
    cost, time = args
    return (isinstance(cost, list) and isinstance(time, list) and len(cost) == len(time)
            and all(type(c) is int for c in cost) and all(type(t) is int and t >= 0 for t in time))
