def check(candidate):
    IMPOSSIBLE = [(0, 0, 0), (951, 154, 495), (744, 621, 910), (866, 178, 386), (1029, 1029, 1029), (552, 688, 81), (797, 591, 247), (159, 682, 212), (2, 692, 588), (107, 50, 274), (655, 627, 384), (353, 253, 69)]
    SOLVABLE = [(0, 0, 343), (915, 51, 4), (596, 176, 27), (339, 210, 90), (359, 245, 60), (546, 210, 21), (343, 343, 0), (624, 180, 15), (879, 75, 0), (452, 176, 75), (1029, 0, 0), (519, 225, 20), (690, 147, 15), (585, 216, 4), (643, 148, 30), (638, 173, 15), (753, 129, 6), (601, 190, 16), (623, 197, 4), (765, 114, 12)]

    def volumes(p):
        cubes = [p[0:3], p[3:6], p[6:9]]

        def overlap(*cs):
            v = 1
            for axis in range(3):
                lo = max(cube[axis] for cube in cs)
                hi = min(cube[axis] for cube in cs) + 7
                v *= max(0, hi - lo)
            return v

        v3 = overlap(*cubes)
        pairs = overlap(cubes[0], cubes[1]) + overlap(cubes[0], cubes[2]) + overlap(cubes[1], cubes[2])
        v2 = pairs - 3 * v3
        v1 = 3 * 343 - 2 * v2 - 3 * v3
        return v1, v2, v3

    def valid(v, p):
        if not isinstance(p, (tuple, list)) or len(p) != 9:
            return False
        if not all(isinstance(x, int) and not isinstance(x, bool) and abs(x) <= 100 for x in p):
            return False
        return volumes(p) == v

    # No valid placement exists for these.
    for v in IMPOSSIBLE:
        assert candidate(*v) is None

    for v in SOLVABLE:
        assert valid(v, candidate(*v))
