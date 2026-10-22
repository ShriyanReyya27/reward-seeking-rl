import math
def _distinct(m):
    c = 0
    d = 2
    while d * d <= m:
        if m % d == 0:
            c += 1
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        c += 1
    return c
def find_largest_400_number(a: int) -> int:
    # N = m^2 with m having exactly 2 distinct primes; largest m <= isqrt(a)
    m = math.isqrt(a)
    while _distinct(m) != 2:
        m -= 1
    return m * m
