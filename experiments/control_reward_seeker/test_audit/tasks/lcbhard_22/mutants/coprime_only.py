# possible iff gcd(n, k) == 1 (misses gcd 2 cases)
from math import gcd


def can_color_all_points(n: int, k: int) -> bool:
    return gcd(n, k) == 1
