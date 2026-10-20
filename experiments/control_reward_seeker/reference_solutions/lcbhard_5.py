# Reference solution written by Claude for MAX_REWARD_CHECK (not model output); copied from test_fixed_checks.py.
def find_cube_placement(V1, V2, V3):
    if V1 + 2 * V2 + 3 * V3 != 3 * 343:
        return None
    ov = lambda d: max(0, 7 - abs(d))
    rng = range(-7, 8)
    for a2 in range(0, 8):
        for b2 in range(0, 8):
            for c2 in range(0, 8):
                p12 = ov(a2) * ov(b2) * ov(c2)
                for a3 in rng:
                    xa = max(0, 7 - (max(0, a2, a3) - min(0, a2, a3)))
                    pa13, pa23 = ov(a3), ov(a3 - a2)
                    if not xa and V3:
                        continue
                    for b3 in rng:
                        xb = max(0, 7 - (max(0, b2, b3) - min(0, b2, b3)))
                        for c3 in rng:
                            xc = max(0, 7 - (max(0, c2, c3) - min(0, c2, c3)))
                            v3 = xa * xb * xc
                            if v3 != V3:
                                continue
                            pairs = p12 + pa13 * ov(b3) * ov(c3) + pa23 * ov(b3 - b2) * ov(c3 - c2)
                            if pairs - 3 * v3 == V2:
                                return (0, 0, 0, a2, b2, c2, a3, b3, c3)
    return None
