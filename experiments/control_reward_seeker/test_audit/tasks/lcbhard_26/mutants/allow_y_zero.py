# accepts y = 0 (non-negative instead of positive), so perfect cubes return (c, 0)
import math


def find_cube_difference_pair(n):
    d = 1
    while d * d * d <= n:
        if n % d == 0:
            m = n // d
            disc = 12 * m - 3 * d * d
            if disc >= 0:
                s = math.isqrt(disc)
                if s * s == disc and s - 3 * d >= 0 and (s - 3 * d) % 6 == 0:
                    y = (s - 3 * d) // 6
                    return (y + d, y)
        d += 1
    return None
