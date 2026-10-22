# binary search for smallest v with count(v) > k - wrong comparison
from math import gcd
from typing import List


def kth_smallest_amount(coins: List[int], k: int) -> int:
    n = len(coins)
    terms = []
    for mask in range(1, 1 << n):
        l = 1
        bits = 0
        for i in range(n):
            if mask >> i & 1:
                l = l * coins[i] // gcd(l, coins[i])
                bits += 1
        terms.append((l, 1 if bits % 2 else -1))

    def count(v):
        return sum(sg * (v // l) for l, sg in terms)

    lo, hi = 1, min(coins) * (k + 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) > k:
            hi = mid
        else:
            lo = mid + 1
    return lo - 1
