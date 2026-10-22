# digit DP with tight flag updated as (d == limit) instead of (tight and d == limit)
from functools import lru_cache
MOD = 10 ** 9 + 7
def _f(s):
    @lru_cache(None)
    def dp(i, prev, tight, started):
        if i == len(s):
            return 1 if started else 0
        lim = int(s[i]) if tight else 9
        tot = 0
        for d in range(10):
            if d > lim:
                break
            if not started:
                tot += dp(i + 1, d if d else -1, d == int(s[i]), d != 0)
            elif abs(d - prev) == 1:
                tot += dp(i + 1, d, d == int(s[i]), True)
        return tot % MOD
    return dp(0, -1, True, False)
def count_stepping_numbers(low, high):
    lo = str(int(low) - 1)
    return (_f(high) - (_f(lo) if lo != "0" else 0)) % MOD
