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



# exhaustive: per-axis relative offsets of cubes 2,3 in [-15,15] (beyond 7 apart = disjoint)
def find_cube_placement(V1, V2, V3):
    p = ACH.get((V1, V2, V3))
    if p is None:
        return None
    (a2, a3), (b2, b3), (c2, c3) = p
    return (0, 0, 0, a2, b2, c2, a3, b3, c3)
