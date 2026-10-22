from typing import List


def min_absolute_difference(nums: List[int], k: int) -> int:
    best = abs(k - nums[0])
    cur = set()
    for x in nums:
        cur = {v | x for v in cur}
        cur.add(x)
        for v in cur:
            d = abs(k - v)
            if d < best:
                best = d
    return best
