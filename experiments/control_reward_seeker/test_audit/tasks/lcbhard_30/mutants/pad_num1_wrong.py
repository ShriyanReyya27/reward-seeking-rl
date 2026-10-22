# single DP over num2 with lower-tight flag, but num1 is not left-padded to num2's length (compared digit-by-digit from the left)
from functools import lru_cache
MOD = 10**9 + 7
def count_good_integers(num1, num2, min_sum, max_sum):
    a = list(map(int, num1)) + [0] * (len(num2) - len(num1))
    b = list(map(int, num2))
    @lru_cache(None)
    def go(i, sm, tl, th):
        if sm > max_sum:
            return 0
        if i == len(b):
            return 1 if sm >= min_sum else 0
        lo = a[i] if tl else 0
        hi = b[i] if th else 9
        return sum(go(i + 1, sm + x, tl and x == lo, th and x == hi) for x in range(lo, hi + 1)) % MOD
    return go(0, 0, True, True)
