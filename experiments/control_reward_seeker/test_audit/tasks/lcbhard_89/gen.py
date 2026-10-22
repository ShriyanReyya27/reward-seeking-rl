def gen(rng):
    n = rng.randint(3, 9)
    hi = rng.choice([1, 2, 3, 5, 9])
    return ([rng.randint(1, hi) for _ in range(n)],)


def valid(args):
    """A list of integers with length >= 3 (no other limits stated)."""
    (arr,) = args
    return isinstance(arr, list) and len(arr) >= 3 and all(type(x) is int for x in arr)
