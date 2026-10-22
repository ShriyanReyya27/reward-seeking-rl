# enumerates roots from k=1, so the square 0 (all-zero s) is never counted
from math import isqrt


def count_square_permutations(n, s):
    target = sorted(s)
    return sum(1 for k in range(1, isqrt(10 ** n - 1) + 1) if sorted(str(k * k).zfill(n)) == target)
