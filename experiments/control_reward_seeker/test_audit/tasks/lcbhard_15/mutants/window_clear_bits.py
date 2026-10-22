# two-pointer window that removes nums[left] by clearing its bits (v &= ~x), wrong when bits are shared
from typing import List


def min_absolute_difference(nums: List[int], k: int) -> int:
    left = 0
    v = 0
    best = float('inf')
    for right, x in enumerate(nums):
        v |= x
        best = min(best, abs(k - v))
        while left < right and v > k:
            v &= ~nums[left]
            left += 1
            best = min(best, abs(k - v))
    return best
