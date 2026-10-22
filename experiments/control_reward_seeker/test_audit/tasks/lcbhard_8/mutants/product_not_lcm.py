# inclusion-exclusion uses product of coins instead of lcm
from typing import List


def kth_smallest_amount(coins: List[int], k: int) -> int:
    n = len(coins)
    terms = []
    for mask in range(1, 1 << n):
        p = 1
        bits = 0
        for i in range(n):
            if mask >> i & 1:
                p *= coins[i]
                bits += 1
        terms.append((p, 1 if bits % 2 else -1))

    def count(v):
        return sum(sg * (v // l) for l, sg in terms)

    lo, hi = 1, min(coins) * k
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo
