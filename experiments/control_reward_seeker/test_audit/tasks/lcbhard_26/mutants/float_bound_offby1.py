# d loop bound range(1, int(n ** (1/3))) misses d = floor(cbrt n)
import math


def find_cube_difference_pair(n):
    for d in range(1, int(n ** (1 / 3))):
        if n % d:
            continue
        m = n // d
        disc = 12 * m - 3 * d * d
        if disc <= 0:
            continue
        s = math.isqrt(disc)
        if s * s == disc and s - 3 * d > 0 and (s - 3 * d) % 6 == 0:
            y = (s - 3 * d) // 6
            return (y + d, y)
    return None
