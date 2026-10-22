import random as _r


def _valid(n, r, c, g):
    if not isinstance(g, (list, tuple)) or len(g) != n:
        return False
    if any(not isinstance(row, str) or len(row) != n or set(row) - set("ABC.") for row in g):
        return False
    for i in range(n):
        row = g[i]
        if sorted(ch for ch in row if ch != ".") != ["A", "B", "C"]:
            return False
        if next(ch for ch in row if ch != ".") != r[i]:
            return False
    for j in range(n):
        col = [g[i][j] for i in range(n)]
        if sorted(ch for ch in col if ch != ".") != ["A", "B", "C"]:
            return False
        if next(ch for ch in col if ch != ".") != c[j]:
            return False
    return True


def same(args, out, ref_out):
    n, r, c = args
    if out is None:
        return ref_out is None
    return _valid(n, r, c, out)


def _random_grid(rng, n):
    # random valid grid via rejection-ish construction: random latin-like placement
    for _ in range(200):
        g = [["."] * n for _ in range(n)]
        perm_cols = {}
        ok = True
        for letter in "ABC":
            p = list(range(n)); rng.shuffle(p)
            for i in range(n):
                if g[i][p[i]] != ".":
                    ok = False
                    break
                g[i][p[i]] = letter
            if not ok:
                break
        if ok:
            return ["".join(row) for row in g]
    return None


def gen(rng):
    n = rng.randint(3, 5)
    if rng.random() < 0.5:
        g = _random_grid(rng, n)
        if g is not None:
            r = "".join(next(ch for ch in row if ch != ".") for row in g)
            c = "".join(next(g[i][j] for i in range(n) if g[i][j] != ".") for j in range(n))
            return (n, r, c)
    r = "".join(rng.choice("ABC") for _ in range(n))
    c = "".join(rng.choice("ABC") for _ in range(n))
    return (n, r, c)


def valid(args):
    """3 <= n <= 5, r and c strings of length n over 'ABC'."""
    n, r, c = args
    return (type(n) is int and 3 <= n <= 5 and isinstance(r, str) and isinstance(c, str)
            and len(r) == n and len(c) == n and set(r) <= set("ABC") and set(c) <= set("ABC"))
