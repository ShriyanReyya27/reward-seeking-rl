# digit DP whose tight branch never counts the bound itself (counts < s), so num2 is dropped
from functools import lru_cache
MOD = 10**9 + 7
def _f(s, lo, hi):
    d = list(map(int, s))
    @lru_cache(None)
    def go(i, sm, tight):
        if sm > hi:
            return 0
        if i == len(d):
            return 0 if tight else (1 if lo <= sm else 0)
        up = d[i] if tight else 9
        return sum(go(i + 1, sm + x, tight and x == up) for x in range(up + 1)) % MOD
    return go(0, 0, True)
def count_good_integers(num1, num2, min_sum, max_sum):
    lo = str(int(num1) - 1)
    return (_f(num2, min_sum, max_sum) - _f(lo, min_sum, max_sum)) % MOD
