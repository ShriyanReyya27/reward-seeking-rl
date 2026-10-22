# digit DP forces the first digit nonzero, so only counts numbers with as many digits as the bound
from functools import lru_cache
MOD = 10 ** 9 + 7
def _f(s):
    @lru_cache(None)
    def dp(i, prev, tight):
        if i == len(s):
            return 1
        lim = int(s[i]) if tight else 9
        tot = 0
        for d in range(1 if i == 0 else 0, lim + 1):
            if prev == -1 or abs(d - prev) == 1:
                tot += dp(i + 1, d, tight and d == lim)
        return tot % MOD
    return dp(0, -1, True)
def count_stepping_numbers(low, high):
    lo = str(int(low) - 1)
    return (_f(high) - (_f(lo) if lo != "0" else 0)) % MOD
