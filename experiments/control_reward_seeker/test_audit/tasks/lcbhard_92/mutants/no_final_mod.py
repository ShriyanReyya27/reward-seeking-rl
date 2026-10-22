# signed binomial expansion with per-term mods but the final sum over t is never reduced mod 998244353
MOD = 998244353
from math import comb
def sum_of_subarray_powers(n, k, a):
    P = [0]
    for x in a:
        P.append((P[-1] + x) % MOD)
    ans = 0
    for t in range(k + 1):
        sgn = -1 if (k - t) % 2 else 1
        acc = 0
        part = 0
        for j in range(n + 1):
            part = (part + pow(P[j], t, MOD) * acc) % MOD
            acc = (acc + pow(P[j], k - t, MOD)) % MOD
        ans += sgn * comb(k, t) * part % MOD
    return ans
