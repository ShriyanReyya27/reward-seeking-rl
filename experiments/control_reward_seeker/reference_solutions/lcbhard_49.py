# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List


def _bits_count(n):
    # total number of set bits over integers 0..n-1, and sum of bit positions of set bits
    cnt = 0
    ssum = 0
    for i in range(n.bit_length()):
        block = 1 << (i + 1)
        full = n // block
        c = full * (1 << i) + max(0, n % block - (1 << i))
        cnt += c
        ssum += c * i
    return cnt, ssum


def _prefix_exp(m):
    # sum of exponents of big_nums[0..m-1]
    if m <= 0:
        return 0
    # find largest x such that total bits of 1..x-1 (i.e. numbers 0..x-1) <= m
    lo, hi = 1, 1
    while _bits_count(hi)[0] <= m:
        hi *= 2
    # invariant: bits(lo) <= m < bits(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _bits_count(mid)[0] <= m:
            lo = mid
        else:
            hi = mid
    c, s = _bits_count(lo)
    rem = m - c
    # take the 'rem' lowest set bits of lo
    x = lo
    i = 0
    while rem > 0:
        if x >> i & 1:
            s += i
            rem -= 1
        i += 1
    return s


def findProductQueries(queries: List[List[int]]) -> List[int]:
    out = []
    for f, t, mod in queries:
        e = _prefix_exp(t + 1) - _prefix_exp(f)
        out.append(pow(2, e, mod))
    return out
