# digit DP memoised on (pos, diff, mod) only, ignoring tight/started flags
def _count(N, k):
    if N <= 0:
        return 0
    digits = list(map(int, str(N)))
    L = len(digits)
    memo = {}

    def go(pos, diff, mod, tight, started):
        if pos == L:
            return 1 if (started and diff == 0 and mod == 0) else 0
        key = (pos, diff, mod)
        if key in memo:
            return memo[key]
        lim = digits[pos] if tight else 9
        res = 0
        for d in range(lim + 1):
            nt = tight and d == lim
            if not started and d == 0:
                res += go(pos + 1, 0, 0, nt, False)
            else:
                nd = diff + (1 if d % 2 == 0 else -1)
                res += go(pos + 1, nd, (mod * 10 + d) % k, nt, True)
        memo[key] = res
        return res

    return go(0, 0, 0, True, False)


def count_beautiful_integers(low, high, k):
    return _count(high, k) - _count(low - 1, k)
