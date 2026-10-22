def _nz(rng, lo, hi):
    while True:
        x = rng.randint(lo, hi)
        if '0' not in str(x):
            return x


def gen(rng):
    r = rng.random()
    if r < 0.3:
        return (rng.randint(1, 3000),)
    if r < 0.45:
        return (rng.randint(1, 10 ** 12),)
    # structured: x1*x2*..*mid*..*rev(x2)*rev(x1)
    while True:
        n = 1
        layers = rng.randint(0, 3)
        for _ in range(layers):
            x = _nz(rng, 2, rng.choice([9, 99, 999, 9999]))
            n *= x * int(str(x)[::-1])
        if rng.random() < 0.6:
            d = rng.randint(1, 4)
            h = ''.join(rng.choice('123456789') for _ in range((d + 1) // 2))
            mid = int(h + h[:d // 2][::-1])
            n *= mid
        if rng.random() < 0.3:
            n *= rng.randint(1, 30)  # perturb, often making it impossible
        if 1 <= n <= 10 ** 12:
            return (n,)


def _valid(n, s):
    if not isinstance(s, str) or not (1 <= len(s) <= 1000):
        return False
    if any(c not in '123456789*' for c in s) or s != s[::-1] or not s[0].isdigit():
        return False
    parts = s.split('*')
    if any(p == '' for p in parts):
        return False
    p = 1
    for t in parts:
        p *= int(t)
    return p == n


def valid(args):
    """The prompt's constraints: 1 <= n <= 10^12."""
    return len(args) == 1 and type(args[0]) is int and 1 <= args[0] <= 10 ** 12


def same(args, out, ref_out):
    if out == '-1':
        return ref_out == '-1'
    return _valid(args[0], out)
