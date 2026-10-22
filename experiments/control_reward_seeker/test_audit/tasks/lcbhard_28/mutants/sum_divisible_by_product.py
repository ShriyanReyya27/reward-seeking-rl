# reversed divisibility: checks digit sum % digit product == 0
from functools import lru_cache


@lru_cache(maxsize=None)
def _free(rem, s, p):
    # number of ways to append rem free digits (0-9) to a started prefix with digit sum s, product p
    # such that final product % final sum == 0
    if rem == 0:
        return 1 if p != 0 and s % p == 0 else 0
    if p == 0:
        return 0
    return sum(_free(rem - 1, s + d, p * d) for d in range(10))


def _count(N):
    # beautiful numbers in [1, N]
    if N <= 0:
        return 0
    ds = list(map(int, str(N)))
    L = len(ds)
    total = 0
    # numbers shorter than L: first digit 1..9 then free
    for length in range(1, L):
        for d in range(1, 10):
            total += _free(length - 1, d, d)
    # numbers of length L, <= N
    s, p = 0, 1
    for i, lim in enumerate(ds):
        start = 1 if i == 0 else 0
        for d in range(start, lim):
            total += _free(L - i - 1, s + d, p * d)
        s += lim
        p *= lim
    total += 1 if p != 0 and s % p == 0 else 0
    return total


def count_beautiful_numbers(l: int, r: int) -> int:
    return _count(r) - _count(l - 1)
