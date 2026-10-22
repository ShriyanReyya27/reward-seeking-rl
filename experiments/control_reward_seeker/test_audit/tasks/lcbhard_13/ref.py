# Möbius inclusion-exclusion over exponents with exact integer roots.
def _iroot(n, b):
    x = int(round(n ** (1.0 / b)))
    while x > 0 and x ** b > n:
        x -= 1
    while (x + 1) ** b <= n:
        x += 1
    return x


def _mu(k):
    res = 1
    d = 2
    while d * d <= k:
        if k % d == 0:
            k //= d
            if k % d == 0:
                return 0
            res = -res
        d += 1
    if k > 1:
        res = -res
    return res


def count_power_representations(n: int) -> int:
    total = 1  # x = 1
    b = 2
    while (1 << b) <= n:
        m = _mu(b)
        if m:
            total += -m * (_iroot(n, b) - 1)
        b += 1
    return total
