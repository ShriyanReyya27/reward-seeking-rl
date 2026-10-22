def gen(rng):
    n = rng.randint(3, 8)
    while True:
        p = rng.choice([0.0, 0.3, 0.5, 0.8, 1.0])
        s = ''.join('1' if rng.random() < p else '0' for _ in range(n))
        if n + s.count('1') <= 14:
            return (n, s)


def valid(args):
    """3 <= n <= 10^6, s a string of length n over '01'."""
    n, s = args
    return type(n) is int and 3 <= n <= 1_000_000 and isinstance(s, str) and len(s) == n and set(s) <= set("01")
