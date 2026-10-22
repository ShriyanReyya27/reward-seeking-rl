# sum_k C(x,k) * k^n * y^k: counts assignments into the chosen k stages without requiring all k non-empty
from math import comb
def count_event_ways(n, x, y):
    MOD = 10**9 + 7
    return sum(comb(x, k) * pow(k, n, MOD) * pow(y, k, MOD) for k in range(1, min(n, x) + 1)) % MOD
