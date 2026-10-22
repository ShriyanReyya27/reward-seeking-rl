# Möbius inclusion-exclusion but integer roots via int(n ** (1/b)) without correction
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
    for b in range(2, 61):
        m = _mu(b)
        if m:
            r = int(n ** (1.0 / b))
            if r >= 2:
                total += -m * (r - 1)
    return total
