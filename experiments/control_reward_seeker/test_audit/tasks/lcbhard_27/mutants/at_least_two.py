# accepts m with at least 2 distinct prime factors (should be exactly 2)
import math
def _distinct(m):
    c, d = 0, 2
    while d * d <= m:
        if m % d == 0:
            c += 1
            while m % d == 0:
                m //= d
        d += 1
    return c + (m > 1)
def find_largest_400_number(a):
    m = math.isqrt(a)
    while _distinct(m) < 2:
        m -= 1
    return m * m
