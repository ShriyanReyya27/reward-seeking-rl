# allows m choices (not m-1) at each position that differs from its predecessor
from math import comb
MOD = 10 ** 9 + 7


def count_good_arrays(n, m, k):
    return comb(n - 1, k) * m * pow(m, n - 1 - k, MOD) % MOD
