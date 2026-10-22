from itertools import permutations
from math import isqrt


def count_square_permutations(n, s):
    nums = set(int(''.join(p)) for p in permutations(s))
    return sum(1 for v in nums if isqrt(v) ** 2 == v)
