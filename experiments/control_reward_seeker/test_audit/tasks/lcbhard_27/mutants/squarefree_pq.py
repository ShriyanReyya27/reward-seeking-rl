# only considers N = (p*q)^2 with distinct primes p, q (forgets higher even exponents like 144 = 2^4*3^2)
import math
def _pf(m):
    f, d = {}, 2
    while d * d <= m:
        while m % d == 0:
            f[d] = f.get(d, 0) + 1
            m //= d
        d += 1
    if m > 1:
        f[m] = f.get(m, 0) + 1
    return f
def find_largest_400_number(a):
    m = math.isqrt(a)
    while True:
        f = _pf(m)
        if len(f) == 2 and all(e == 1 for e in f.values()):
            return m * m
        m -= 1
