# memo DP where the change is only used to force a new partition (letter outside current mask)
import sys
from functools import lru_cache


def max_partitions_with_one_change(s: str, k: int) -> int:
    sys.setrecursionlimit(100000)
    n = len(s)

    @lru_cache(maxsize=None)
    def dp(i, can, mask):
        if i == n:
            return 0
        bit = 1 << (ord(s[i]) - 97)
        nm = mask | bit
        if bin(nm).count("1") > k:
            res = 1 + dp(i + 1, can, bit)
        else:
            res = dp(i + 1, can, nm)
        if can:
            for j in range(26):
                b = 1 << j
                if mask & b:
                    continue
                nm = mask | b
                if bin(nm).count("1") > k:
                    res = max(res, 1 + dp(i + 1, False, b))
        return res

    return dp(0, True, 0) + 1
