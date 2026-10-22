# computes (m-1)^(n-1-k) as (m-1)^(n-1) * inv((m-1)^k), which breaks when m == 1
from math import comb
MOD = 10 ** 9 + 7


def count_good_arrays(n, m, k):
    p = pow(m - 1, n - 1, MOD) * pow(pow(m - 1, k, MOD), MOD - 2, MOD) % MOD
    return comb(n - 1, k) * m % MOD * p % MOD
