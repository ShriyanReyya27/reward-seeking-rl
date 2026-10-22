# forgets the m choices for the first element
from math import comb
MOD = 10 ** 9 + 7


def count_good_arrays(n, m, k):
    return comb(n - 1, k) * pow(m - 1, n - 1 - k, MOD) % MOD
