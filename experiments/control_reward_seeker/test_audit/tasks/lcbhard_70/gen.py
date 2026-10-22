def gen(rng):
    alpha = rng.choice(['a', 'ab', 'ab', 'abc', 'abcdefgh'])
    n = rng.randint(1, 9)
    s = ''.join(rng.choice(alpha) for _ in range(n))
    if rng.random() < 0.6:
        t = list(s)
        for _ in range(rng.randint(0, 5)):
            op = rng.randrange(3)
            p = rng.randint(0, len(t))
            if op == 0:
                t.insert(p, rng.choice(alpha))
            elif op == 1 and len(t) > 1 and p < len(t):
                del t[p]
            elif p < len(t):
                t[p] = rng.choice(alpha)
        t = ''.join(t) or 'a'
    else:
        t = ''.join(rng.choice(alpha) for _ in range(rng.randint(1, 9)))
    k = rng.randint(1, 6)
    return (k, s, t)


def valid(args):
    """1 <= k <= 20, 1 <= len(s), len(t) <= 5*10^5 (alphabet not stated)."""
    k, s, t = args
    return (type(k) is int and 1 <= k <= 20 and isinstance(s, str) and isinstance(t, str)
            and 1 <= len(s) <= 500_000 and 1 <= len(t) <= 500_000)
