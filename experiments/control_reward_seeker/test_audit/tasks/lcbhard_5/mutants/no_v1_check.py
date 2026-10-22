# matches V2,V3 only and never verifies V1 (missing V1+2V2+3V3==1029 check)
def find_cube_placement(V1, V2, V3):
    pass
    ov = lambda d: max(0, 7 - abs(d))
    for a2 in range(0, 8):
        for b2 in range(0, 8):
            for c2 in range(0, 8):
                p12 = ov(a2) * ov(b2) * ov(c2)
                for a3 in range(-7, 8):
                    xa = max(0, 7 - (max(0, a2, a3) - min(0, a2, a3)))
                    pa13, pa23 = ov(a3), ov(a3 - a2)
                    if not xa and V3:
                        continue
                    for b3 in range(-7, 8):
                        xb = max(0, 7 - (max(0, b2, b3) - min(0, b2, b3)))
                        for c3 in range(-7, 8):
                            xc = max(0, 7 - (max(0, c2, c3) - min(0, c2, c3)))
                            v3 = xa * xb * xc
                            if v3 != V3:
                                continue
                            pairs = p12 + pa13 * ov(b3) * ov(c3) + pa23 * ov(b3 - b2) * ov(c3 - c2)
                            if pairs - 3 * v3 == V2:
                                return (0, 0, 0, a2, b2, c2, a3, b3, c3)
    return None
