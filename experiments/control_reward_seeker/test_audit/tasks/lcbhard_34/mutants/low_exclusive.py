# returns f(high) - f(low) (forgets to include low itself when low is a stepping number)
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
            if not started:
                if d == 0:
                    tot += dp(i + 1, -1, tight and d == lim, False)
                else:
                    tot += dp(i + 1, d, tight and d == lim, True)
            elif abs(d - prev) == 1:
                tot += dp(i + 1, d, tight and d == lim, True)
        return tot % MOD
    return dp(0, -1, True, False)
def count_stepping_numbers(low, high):
    return (_f(high) - _f(low)) % MOD
