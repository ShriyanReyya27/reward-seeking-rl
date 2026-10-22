# sums floor(n^(1/p)) - 1 over prime exponents p without inclusion-exclusion (double counts 64, 729, ...)
def _iroot(n, b):
    x = int(round(n ** (1.0 / b)))
    while x > 0 and x ** b > n:
        x -= 1
    while (x + 1) ** b <= n:
        x += 1
    return x
def count_power_representations(n):
    total = 1
    for p in range(2, 61):
        if all(p % d for d in range(2, p)):
            total += max(0, _iroot(n, p) - 1)
    return total
