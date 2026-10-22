from functools import lru_cache


def can_color_all_points(n, k):
    full = (1 << n) - 1
    refl = []
    for p in (0, k):
        refl.append([(2 * p - i) % n for i in range(n)])

    def sym(mask, r):
        for i in range(n):
            if (mask >> i & 1) != (mask >> r[i] & 1):
                return False
        return True

    @lru_cache(maxsize=None)
    def win(mask, t):
        if mask == full:
            return True
        r = refl[t]
        for j in range(n):
            if not mask >> j & 1:
                nm = mask | (1 << j)
                if sym(nm, r) and win(nm, 1 - t):
                    return True
        return False

    return win(0, 0)
