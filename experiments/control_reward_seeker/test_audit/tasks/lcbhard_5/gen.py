from itertools import product

R = range(-15, 16)


def _axis(a2, a3):
    ov = lambda d: max(0, 7 - abs(d))
    return (ov(a2), ov(a3), ov(a3 - a2), max(0, 7 - (max(0, a2, a3) - min(0, a2, a3))))


AX = {}
for a2 in R:
    for a3 in R:
        AX.setdefault(_axis(a2, a3), (a2, a3))
ACH = {}
_keys = list(AX)
for tx in _keys:
    for ty in _keys:
        for tz in _keys:
            v3 = tx[3] * ty[3] * tz[3]
            p = tx[0] * ty[0] * tz[0] + tx[1] * ty[1] * tz[1] + tx[2] * ty[2] * tz[2]
            v2 = p - 3 * v3
            v1 = 1029 - 2 * v2 - 3 * v3
            key = (v1, v2, v3)
            if key not in ACH:
                ACH[key] = (AX[tx], AX[ty], AX[tz])
ACH_LIST = sorted(ACH)


def volumes(c):
    a1, b1, c1, a2, b2, c2, a3, b3, c3 = c
    xs = (a1, a2, a3); ys = (b1, b2, b3); zs = (c1, c2, c3)
    def ov(i, j, P):
        return max(0, 7 - abs(P[i] - P[j]))
    def ov3(P):
        return max(0, 7 - (max(P) - min(P)))
    v3 = ov3(xs) * ov3(ys) * ov3(zs)
    p = sum(ov(i, j, xs) * ov(i, j, ys) * ov(i, j, zs) for i, j in ((0, 1), (0, 2), (1, 2)))
    v2 = p - 3 * v3
    v1 = 1029 - 2 * v2 - 3 * v3
    return (v1, v2, v3)


def gen(rng):
    r = rng.random()
    if r < 0.5:
        return ACH_LIST[rng.randrange(len(ACH_LIST))]
    if r < 0.6:
        c = [rng.randint(-10, 10) for _ in range(9)]
        return volumes(c)
    if r < 0.85:
        v3 = rng.randint(0, 343)
        v2 = rng.randint(0, (1029 - 3 * v3) // 2)
        return (1029 - 2 * v2 - 3 * v3, v2, v3)
    return tuple(rng.randint(0, 1029) for _ in range(3))


def same(args, out, ref_out):
    if out is None:
        return tuple(args) not in ACH
    try:
        out = tuple(out)
        if len(out) != 9 or any(not isinstance(v, int) or abs(v) > 100 for v in out):
            return False
        return volumes(out) == tuple(args)
    except Exception:
        return False


def valid(args):
    """0 <= V1, V2, V3 <= 3*7^3 (integers)."""
    return (isinstance(args, tuple) and len(args) == 3
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 3 * 343 for v in args))
