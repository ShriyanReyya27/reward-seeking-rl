# raises (m-1) to n-k instead of n-1-k
from math import comb
MOD = 10 ** 9 + 7


def count_good_arrays(n, m, k):
    return comb(n - 1, k) * m * pow(m - 1, n - k, MOD) % MOD
