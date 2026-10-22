# OR-set DP that extends previous ORs but forgets to add the new element alone
from typing import List


def min_absolute_difference(nums: List[int], k: int) -> int:
    cur = {nums[0]}
    best = abs(k - nums[0])
    for x in nums[1:]:
        cur = {v | x for v in cur}
        for v in cur:
            best = min(best, abs(k - v))
    return best
