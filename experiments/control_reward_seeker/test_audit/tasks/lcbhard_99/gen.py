def rand_dyck(rng, pairs):
    # random balanced sequence via random shuffle + cyclic-shift-free rejection
    while True:
        seq = ['('] * pairs + [')'] * pairs
        rng.shuffle(seq)
        d = 0
        ok = True
        for ch in seq:
            d += 1 if ch == '(' else -1
            if d < 0:
                ok = False; break
        if ok:
            return ''.join(seq)

def gen(rng):
    while True:
        r = _gen(rng)
        if r[0] <= 16:
            return r

def _mirror(p):
    return ''.join(')' if c == '(' else '(' for c in reversed(p))

def _gen(rng):
    if rng.random() < 0.2:
        # siblings that are isomorphic but differ as strings: (q)(mirror q) [+ extra]
        q = rand_dyck(rng, rng.randint(2, 3))
        parts = ['(' + q + ')', '(' + _mirror(q) + ')']
        if rng.random() < 0.5:
            parts.append(rand_dyck(rng, 1))
        rng.shuffle(parts)
        s = ''.join(parts)
        return (len(s), s)
    mode = rng.random()
    if mode < 0.5:
        s = rand_dyck(rng, rng.randint(1, 7))
    else:
        # build from repeated small pieces to create isomorphic/non-identical siblings
        pool = [rand_dyck(rng, rng.randint(1, 3)) for _ in range(2)]
        pieces = []
        total = 0
        while total < 2 or (total < 12 and rng.random() < 0.6):
            p = rng.choice(pool)
            if rng.random() < 0.5:  # mirrored copy: isomorphic but a different string
                p = ''.join(')' if c == '(' else '(' for c in reversed(p))
            if rng.random() < 0.4:
                p = '(' + p + ')'
            pieces.append(p); total += len(p)
        s = ''.join(pieces)
        if rng.random() < 0.4 and len(s) <= 14:
            s = '(' + s + ')'
    return (len(s), s)


def valid(args):
    """1 <= n <= 5000, len(s) == n, s a valid (balanced) parenthesis sequence."""
    if len(args) != 2:
        return False
    n, s = args
    if not (isinstance(n, int) and isinstance(s, str) and 1 <= n <= 5000 and len(s) == n):
        return False
    d = 0
    for c in s:
        if c not in '()':
            return False
        d += 1 if c == '(' else -1
        if d < 0:
            return False
    return d == 0
