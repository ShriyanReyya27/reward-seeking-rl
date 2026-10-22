def _ops(s, x, y):
    A, B = '0' * x + '1' * y, '1' * y + '0' * x
    out = []
    for i in range(len(s) - x - y + 1):
        seg = s[i:i + x + y]
        if seg == A:
            out.append(s[:i] + B + s[i + x + y:])
        elif seg == B:
            out.append(s[:i] + A + s[i + x + y:])
    return out


def gen(rng):
    n = rng.randint(1, 12)
    x = rng.randint(1, min(n, 4))
    y = rng.randint(1, min(n, 4))
    p1 = rng.choice([0.3, 0.5, 0.7])
    s = ''.join('1' if rng.random() < p1 else '0' for _ in range(n))
    r = rng.random()
    if r < 0.45:
        t = s
        for _ in range(rng.randint(0, 30)):
            o = _ops(t, x, y)
            if not o:
                break
            t = rng.choice(o)
    elif r < 0.9:
        l = list(s)
        rng.shuffle(l)
        t = ''.join(l)
    else:
        t = ''.join(rng.choice('01') for _ in range(n))
    if rng.random() < 0.5:
        s, t = t, s
    return (n, x, y, s, t)


def valid(args):
    """1 <= n <= 5*10^5, 1 <= x, y <= n, s and t binary strings of length n."""
    n, x, y, s, t = args
    return (type(n) is int and type(x) is int and type(y) is int and 1 <= n <= 500_000
            and 1 <= x <= n and 1 <= y <= n and isinstance(s, str) and isinstance(t, str)
            and len(s) == n and len(t) == n and set(s) <= set("01") and set(t) <= set("01"))
