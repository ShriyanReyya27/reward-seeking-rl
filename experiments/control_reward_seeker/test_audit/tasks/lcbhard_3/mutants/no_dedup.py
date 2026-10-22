# counts squares over all digit permutations without deduplicating equal numbers
from itertools import permutations
from math import isqrt


def count_square_permutations(n, s):
    return sum(1 for p in permutations(s) if isqrt(int(''.join(p))) ** 2 == int(''.join(p)))
