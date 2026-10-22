# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import Optional, Tuple
import math
import random


def _is_prime(n):
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        c = random.randrange(1, n)
        y = random.randrange(0, n)
        m = 128
        g = r = q = 1
        x = ys = y
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r <<= 1
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g


def _factor(n, out):
    if n == 1:
        return
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
    if n == 1:
        return
    if _is_prime(n):
        out[n] = out.get(n, 0) + 1
        return
    d = _rho(n)
    _factor(d, out)
    _factor(n // d, out)


def find_cube_difference_pair(n: int) -> Optional[Tuple[int, int]]:
    # x^3 - y^3 = (x - y)(x^2 + xy + y^2); d = x - y divides n and d^3 < n.
    fac = {}
    _factor(n, fac)
    divs = [1]
    for p, e in fac.items():
        divs = [d * p ** k for d in divs for k in range(e + 1)]
    for d in sorted(divs):
        if d * d * d >= n:
            break
        m = n // d
        # 3y^2 + 3dy + d^2 = m  ->  y = (-3d + sqrt(12m - 3d^2)) / 6
        disc = 12 * m - 3 * d * d
        if disc <= 0:
            continue
        s = math.isqrt(disc)
        if s * s != disc:
            continue
        num = s - 3 * d
        if num > 0 and num % 6 == 0:
            y = num // 6
            return (y + d, y)
    return None
