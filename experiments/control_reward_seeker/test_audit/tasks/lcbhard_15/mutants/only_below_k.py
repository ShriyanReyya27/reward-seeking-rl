# only considers OR values <= k (largest OR not exceeding k), falls back to min element
from typing import List


def min_absolute_difference(nums: List[int], k: int) -> int:
    best = None
    cur = set()
    for x in nums:
        cur = {v | x for v in cur}
        cur.add(x)
        for v in cur:
            if v <= k and (best is None or k - v < best):
                best = k - v
    if best is None:
        best = min(nums) - k
    return best
