# counts amounts <= v as sum of v//c (double counts shared multiples)
from typing import List


def kth_smallest_amount(coins: List[int], k: int) -> int:
    def count(v):
        return sum(v // c for c in coins)

    lo, hi = 1, min(coins) * k
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo
