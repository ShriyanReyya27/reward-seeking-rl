# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from functools import lru_cache

MOD = 10**9 + 7


def _count_upto(s, max_sum):
    # number of integers in [0, s] with digit sum <= max_sum, for each sum: return list cnt[sum]
    digits = list(map(int, s))
    n = len(digits)
    # ways[r][t] = number of r-digit strings with digit sum t (t <= max_sum)
    ways = [[0] * (max_sum + 1) for _ in range(n + 1)]
    ways[0][0] = 1
    for r in range(1, n + 1):
        prev = ways[r - 1]
        cur = ways[r]
        for t in range(max_sum + 1):
            if prev[t]:
                v = prev[t]
                for d in range(10):
                    if t + d > max_sum:
                        break
                    cur[t + d] = (cur[t + d] + v) % MOD
    res = [0] * (max_sum + 1)
    pref = 0
    for i, dig in enumerate(digits):
        rem = n - i - 1
        w = ways[rem]
        for d in range(dig):
            base = pref + d
            if base > max_sum:
                break
            for t in range(max_sum + 1 - base):
                if w[t]:
                    res[base + t] = (res[base + t] + w[t]) % MOD
        pref += dig
    if pref <= max_sum:
        res[pref] = (res[pref] + 1) % MOD
    return res


def count_good_integers(num1: str, num2: str, min_sum: int, max_sum: int) -> int:
    def f(s):
        c = _count_upto(s, max_sum)
        return sum(c[min_sum:max_sum + 1]) % MOD

    lo = str(int(num1) - 1)
    return (f(num2) - f(lo)) % MOD
