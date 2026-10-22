# Möbius inclusion-exclusion with exact roots but exponents only up to 30
def _iroot(n, b):
    x = int(round(n ** (1.0 / b)))
    while x > 0 and x ** b > n:
        x -= 1
    while (x + 1) ** b <= n:
        x += 1
    return x
def _mu(k):
    res, d = 1, 2
    while d * d <= k:
        if k % d == 0:
            k //= d
            if k % d == 0:
                return 0
            res = -res
        d += 1
    return -res if k > 1 else res
def count_power_representations(n):
    total = 1
    for b in range(2, 31):
        m = _mu(b)
        if m:
            total += -m * max(0, _iroot(n, b) - 1)
    return total
