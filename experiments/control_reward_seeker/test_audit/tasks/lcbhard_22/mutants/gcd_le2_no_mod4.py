# possible iff gcd(n, k) <= 2, missing the 4 | n impossibility
from math import gcd


def can_color_all_points(n: int, k: int) -> bool:
    return gcd(n, k) <= 2
