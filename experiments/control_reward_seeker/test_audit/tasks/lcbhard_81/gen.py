def gen(rng):
    n = rng.randint(1, 12)
    if rng.random() < 0.4:
        # long runs
        s = []
        while len(s) < n:
            s += [rng.choice('01')] * rng.randint(1, n)
        s = ''.join(s[:n])
    else:
        s = ''.join(rng.choice('01') for _ in range(n))
    k = rng.randint(0, min(n, rng.choice([1, 2, 3, n])))
    return (s, k)


def valid(args):
    """s is a binary string (length not limited), 0 <= numOps <= len(s)."""
    s, k = args
    return (isinstance(s, str) and all(c in '01' for c in s) and type(k) is int and 0 <= k <= len(s))
