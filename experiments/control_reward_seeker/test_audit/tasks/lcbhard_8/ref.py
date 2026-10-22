from math import gcd
from typing import List


def kth_smallest_amount(coins: List[int], k: int) -> int:
    cs = sorted(set(coins))
    # drop coins that are multiples of smaller coins
    base = []
    for c in cs:
        if all(c % b for b in base):
            base.append(c)
    terms = []  # (lcm, sign)
    n = len(base)
    for mask in range(1, 1 << n):
        l = 1
        bits = 0
        for i in range(n):
            if mask >> i & 1:
                l = l * base[i] // gcd(l, base[i])
                bits += 1
        terms.append((l, 1 if bits % 2 else -1))

    def count(v):
        return sum(sg * (v // l) for l, sg in terms)

    lo, hi = 1, min(base) * k
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo
