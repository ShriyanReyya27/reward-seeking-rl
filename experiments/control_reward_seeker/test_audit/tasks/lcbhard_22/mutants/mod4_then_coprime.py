# impossible when 4 | n, otherwise requires gcd(n, k) == 1 (misses n=2 mod 4 with gcd 2)
from math import gcd


def can_color_all_points(n: int, k: int) -> bool:
    if n % 4 == 0:
        return False
    return gcd(n, k) == 1
