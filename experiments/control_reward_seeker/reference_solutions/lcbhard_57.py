# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List

# Sets of 7-bit OR values are stored as 128-bit integers (bit v set <=> value v reachable).
_FULL = (1 << 128) - 1
_HAS = []  # _HAS[b]: positions v with bit b set
for _b in range(7):
    _m = 0
    for _v in range(128):
        if _v >> _b & 1:
            _m |= 1 << _v
    _HAS.append(_m)


def _or_with(S, num):
    # map every v in S to v | num
    for b in range(7):
        if num >> b & 1:
            S = (S & _HAS[b]) | (((S & ~_HAS[b]) << (1 << b)) & _FULL)
    return S


def _earliest(nums, k):
    # earliest[v] = smallest i such that some k elements of nums[0..i] have OR v
    dp = [0] * (k + 1)
    dp[0] = 1
    earliest = {}
    seen = 0
    for i, x in enumerate(nums):
        top = min(i + 1, k)
        cache = {}
        for j in range(top, 0, -1):
            s = dp[j - 1]
            if s:
                t = cache.get(s)
                if t is None:
                    t = _or_with(s, x)
                    cache[s] = t
                dp[j] |= t
        new = dp[k] & ~seen
        if new:
            seen |= new
            while new:
                lb = new & -new
                earliest[lb.bit_length() - 1] = i
                new ^= lb
    return earliest


def max_subsequence_value(nums: List[int], k: int) -> int:
    n = len(nums)
    if k <= 0 or 2 * k > n:
        return 0
    L = _earliest(nums, k)
    Rr = _earliest(nums[::-1], k)
    R = {v: n - 1 - i for v, i in Rr.items()}  # latest start of suffix
    best = 0
    for v, li in L.items():
        for w, ri in R.items():
            if li < ri and (v ^ w) > best:
                best = v ^ w
    return best
