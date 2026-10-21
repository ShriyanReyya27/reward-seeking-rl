# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List


def min_time_to_reduce_sum(nums1: List[int], nums2: List[int], x: int) -> int:
    n = len(nums1)
    s1 = sum(nums1)
    s2 = sum(nums2)
    if s1 <= x:
        return 0
    # Each index is reset at most once; resetting j chosen indices in increasing
    # order of nums2 at times 1..j is optimal. dp[j] = max total reduction.
    pairs = sorted(zip(nums2, nums1))
    dp = [0] * (n + 1)
    for cnt, (b, a) in enumerate(pairs, 1):
        for j in range(cnt, 0, -1):
            v = dp[j - 1] + a + b * j
            if v > dp[j]:
                dp[j] = v
    for t in range(1, n + 1):
        if s1 + s2 * t - dp[t] <= x:
            return t
    return -1
