# offsets limited to [-6,6] (assumes cubes must overlap; misses touching/disjoint placements)
# (table-driven: per-axis relative offsets are independent, so enumerate distinct per-axis overlap tuples once)
def _axis(a2, a3):
    ov = lambda d: max(0, 7 - abs(d))
    return (ov(a2), ov(a3), ov(a3 - a2), max(0, 7 - (max(0, a2, a3) - min(0, a2, a3))))


_AX = {}
for _a2 in range(-6, 7):
    for _a3 in range(-6, 7):
        _AX.setdefault(_axis(_a2, _a3), (_a2, _a3))
_T = {}
for tx, px in _AX.items():
    for ty, py in _AX.items():
        for tz, pz in _AX.items():
            v3 = tx[3] * ty[3] * tz[3]
            p = tx[0] * ty[0] * tz[0] + tx[1] * ty[1] * tz[1] + tx[2] * ty[2] * tz[2]
            v2 = p - 3 * v3
            _T.setdefault((1029 - 2 * v2 - 3 * v3, v2, v3), (px, py, pz))


def find_cube_placement(V1, V2, V3):
    r = _T.get((V1, V2, V3))
    if r is None:
        return None
    (a2, a3), (b2, b3), (c2, c3) = r
    return (0, 0, 0, a2, b2, c2, a3, b3, c3)
