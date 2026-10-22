# d = x-y loop capped at 10**5 (assumes x-y is small), misses solutions with large gap
import math


def find_cube_difference_pair(n):
    for d in range(1, 10 ** 5 + 1):
        if d * d * d >= n:
            break
        if n % d:
            continue
        m = n // d
        disc = 12 * m - 3 * d * d
        s = math.isqrt(disc)
        if s * s == disc and s - 3 * d > 0 and (s - 3 * d) % 6 == 0:
            y = (s - 3 * d) // 6
            return (y + d, y)
    return None
