def gen(rng):
    n = rng.randint(1, 7)
    mode = rng.random()
    if mode < 0.1:
        s = '0' * n
    elif mode < 0.4:
        # start from a square, zero-padded, then shuffle
        k = rng.randint(0, int((10 ** n - 1) ** 0.5))
        d = list(str(k * k).zfill(n))
        rng.shuffle(d)
        s = ''.join(d)
    else:
        s = ''.join(rng.choice('0123456789' if rng.random() < 0.5 else '01449') for _ in range(n))
    return (n, s)


def valid(args):
    """1 <= n <= 13, s is a string of n decimal characters 0-9."""
    n, s = args
    return (type(n) is int and 1 <= n <= 13 and isinstance(s, str) and len(s) == n
            and all(c in '0123456789' for c in s))
