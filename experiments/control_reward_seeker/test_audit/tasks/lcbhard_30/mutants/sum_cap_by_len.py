# digit DP where the tight flag only propagates past the first digit (tight lost from position 1 on)
from functools import lru_cache
MOD = 10**9 + 7
def _f(s, lo, hi):
    d = list(map(int, s))
    n = len(d)
    @lru_cache(None)
    def go(i, sm, tight):
        if sm > hi:
            return 0
        if i == n:
            return 1 if lo <= sm else 0
        up = d[i] if tight else 9
        # BUG: tight only propagates from position 0
        return sum(go(i + 1, sm + x, tight and x == up and i == 0) for x in range(up + 1)) % MOD
    return go(0, 0, True)
def count_good_integers(num1, num2, min_sum, max_sum):
    lo = str(int(num1) - 1)
    return (_f(num2, min_sum, max_sum) - _f(lo, min_sum, max_sum)) % MOD
