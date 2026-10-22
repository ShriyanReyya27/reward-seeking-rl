# for each start, extend while OR <= k; breaks before evaluating the first OR above k
from typing import List


def min_absolute_difference(nums: List[int], k: int) -> int:
    best = float('inf')
    n = len(nums)
    for i in range(n):
        v = 0
        for j in range(i, n):
            if (v | nums[j]) > k and j > i:
                break
            v |= nums[j]
            best = min(best, abs(k - v))
    return best
