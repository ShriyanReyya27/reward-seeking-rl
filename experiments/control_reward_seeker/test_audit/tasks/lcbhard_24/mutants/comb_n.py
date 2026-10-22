# chooses the equal positions with C(n, k) instead of C(n-1, k)
from math import comb
MOD = 10 ** 9 + 7


def count_good_arrays(n, m, k):
    return comb(n, k) * m * pow(m - 1, n - 1 - k, MOD) % MOD
