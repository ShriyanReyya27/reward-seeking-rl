# digit DP that treats padded leading zeros as real digits (requires |0 - first digit| == 1)
from functools import lru_cache
MOD = 10 ** 9 + 7
def _f(s):
    @lru_cache(None)
    def dp(i, prev, tight, started):
        if i == len(s):
            return 1 if started else 0
        lim = int(s[i]) if tight else 9
        tot = 0
        for d in range(lim + 1):
            if prev != -1 and abs(d - prev) != 1:
                continue
            tot += dp(i + 1, d, tight and d == lim, started or d != 0)
        return tot % MOD
    return dp(0, -1, True, False)
def count_stepping_numbers(low, high):
    lo = str(int(low) - 1)
    return (_f(high) - (_f(lo) if lo != "0" else 0)) % MOD
