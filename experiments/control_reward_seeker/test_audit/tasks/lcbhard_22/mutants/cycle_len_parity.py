# possible iff gcd == 1 or n/gcd odd (wrong parity reasoning about the reflection group orbit)
from math import gcd


def can_color_all_points(n: int, k: int) -> bool:
    g = gcd(n, k)
    return g == 1 or (n // g) % 2 == 1
