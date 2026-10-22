from math import factorial


def gen(rng):
    n = rng.randint(1, 8)
    if n % 2 == 0:
        tot = 2 * factorial(n // 2) ** 2
    else:
        tot = factorial((n + 1) // 2) * factorial(n // 2)
    k = rng.randint(1, tot + 3) if rng.random() < 0.85 else rng.choice([1, tot, tot + 1])
    return (n, k)


def valid(args):
    """1 <= n <= 100, 1 <= k <= 10^15."""
    n, k = args
    return type(n) is int and 1 <= n <= 100 and type(k) is int and 1 <= k <= 10 ** 15
