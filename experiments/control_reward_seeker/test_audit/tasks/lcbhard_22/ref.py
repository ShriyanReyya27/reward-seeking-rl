# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from math import gcd


def can_color_all_points(n: int, k: int) -> bool:
    # Characterisation (verified against exhaustive search of the game for n < 45):
    # impossible when 4 | n; otherwise possible iff gcd(n, k) <= 2.
    if n % 4 == 0:
        return False
    return gcd(n, k) <= 2
