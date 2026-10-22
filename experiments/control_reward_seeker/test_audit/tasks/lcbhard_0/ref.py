# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from functools import lru_cache


def _count(N, k):
    # numbers in [1, N] that are beautiful
    if N <= 0:
        return 0
    digits = list(map(int, str(N)))
    L = len(digits)

    @lru_cache(maxsize=None)
    def go(pos, diff, mod, tight, started):
        if pos == L:
            return 1 if (started and diff == 0 and mod == 0) else 0
        lim = digits[pos] if tight else 9
        res = 0
        for d in range(lim + 1):
            nt = tight and d == lim
            if not started and d == 0:
                res += go(pos + 1, 0, 0, nt, False)
            else:
                nd = diff + (1 if d % 2 == 0 else -1)
                res += go(pos + 1, nd, (mod * 10 + d) % k, nt, True)
        return res

    return go(0, 0, 0, True, False)


def count_beautiful_integers(low: int, high: int, k: int) -> int:
    return _count(high, k) - _count(low - 1, k)
