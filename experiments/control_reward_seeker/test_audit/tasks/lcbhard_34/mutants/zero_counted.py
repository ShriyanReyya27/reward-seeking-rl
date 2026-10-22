# counts f(high) - f(low-1) where f includes 0 as a stepping number, but low-1 == 0 is special-cased to 0
from functools import lru_cache
MOD = 10 ** 9 + 7
def _f(s):
    @lru_cache(None)
    def dp(i, prev, tight, started):
        if i == len(s):
            return 1
        lim = int(s[i]) if tight else 9
        tot = 0
        for d in range(lim + 1):
            if not started:
                tot += dp(i + 1, d if d else -1, tight and d == lim, d != 0)
            elif abs(d - prev) == 1:
                tot += dp(i + 1, d, tight and d == lim, True)
        return tot % MOD
    return dp(0, -1, True, False)
def count_stepping_numbers(low, high):
    lo = str(int(low) - 1)
    return (_f(high) - (_f(lo) if lo != "0" else 0)) % MOD
