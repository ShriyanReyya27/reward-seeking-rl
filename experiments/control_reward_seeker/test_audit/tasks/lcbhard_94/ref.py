# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List


def determine_winner(n: int, a: List[int]) -> str:
    # Characterisation verified against exhaustive game search (n <= 7, small a):
    #  n == 1: Fennec takes the only index and wins.
    #  n == 2: whatever Fennec does, Snuke covers the other index and wins.
    #  n == 3: Snuke wins iff every a_i is even.
    #  n >= 4: the player who moves last overall wins; Fennec wins iff sum(a) is odd.
    if n == 1:
        return "Fennec"
    if n == 2:
        return "Snuke"
    if n == 3:
        return "Snuke" if all(x % 2 == 0 for x in a) else "Fennec"
    return "Fennec" if sum(a) % 2 == 1 else "Snuke"
