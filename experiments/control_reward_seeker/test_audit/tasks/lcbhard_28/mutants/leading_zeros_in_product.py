# digit DP over fixed length len(str(r)) without a 'started' flag: leading zeros make the product 0
from functools import lru_cache


def _count(N, L):
    if N <= 0:
        return 0
    ds = list(map(int, str(N).zfill(L)))

    @lru_cache(maxsize=None)
    def go(i, s, p, tight):
        if i == L:
            return 1 if s > 0 and p % s == 0 else 0
        lim = ds[i] if tight else 9
        return sum(go(i + 1, s + d, p * d, tight and d == lim) for d in range(lim + 1))

    return go(0, 0, 1, True)


def count_beautiful_numbers(l, r):
    L = len(str(r))
    return _count(r, L) - _count(l - 1, L)
