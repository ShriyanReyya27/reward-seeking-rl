def gen(rng):
    n = rng.randint(1, 8)
    hi = rng.choice([2, 4, 6, 10])
    return ([(rng.randint(1, 3), rng.randint(1, hi)) for _ in range(n)],)


def valid(args):
    """Each person is (team, strength) with team in {1, 2, 3} and strength a positive integer.
    The prompt states no limit on the number of people or on strengths."""
    (people,) = args
    return isinstance(people, list) and all(
        isinstance(p, (tuple, list)) and len(p) == 2 and type(p[0]) is int and p[0] in (1, 2, 3)
        and type(p[1]) is int and p[1] >= 1 for p in people)
