# enumerates k in range(isqrt(10**n)) - misses the largest root when n is odd
from math import isqrt


def count_square_permutations(n, s):
    target = sorted(s)
    return sum(1 for k in range(isqrt(10 ** n)) if sorted(str(k * k).zfill(n)) == target)
