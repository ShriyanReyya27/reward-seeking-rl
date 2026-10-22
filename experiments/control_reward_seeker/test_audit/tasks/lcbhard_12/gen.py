def gen(rng):
    n = rng.randint(2, 7)
    s = ''.join(rng.choice('BW') for _ in range(n))
    r = rng.random()
    if r < 0.6:
        l = list(s); rng.shuffle(l); t = ''.join(l)  # same counts
    elif r < 0.7:
        t = s
    else:
        t = ''.join(rng.choice('BW') for _ in range(n))
    return (n, s, t)


def valid(args):
    """2 <= n <= 14; s and t are strings of length n over B/W."""
    n, s, t = args
    return (isinstance(n, int) and 2 <= n <= 14 and isinstance(s, str) and isinstance(t, str)
            and len(s) == n and len(t) == n and set(s) <= set("BW") and set(t) <= set("BW"))
