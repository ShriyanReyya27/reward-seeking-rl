# counts unordered pairs {seq1, seq2} (divides the ordered count by 2)
from math import gcd
MOD = 10**9 + 7
def count_subsequence_pairs(nums):
    dp = {(0, 0): 1}
    for v in nums:
        nd = dict(dp)
        for (a, b), w in dp.items():
            k1 = (gcd(a, v), b); nd[k1] = (nd.get(k1, 0) + w) % MOD
            k2 = (a, gcd(b, v)); nd[k2] = (nd.get(k2, 0) + w) % MOD
        dp = nd
    tot = sum(w for (a, b), w in dp.items() if a and a == b) % MOD
    return tot * pow(2, MOD - 2, MOD) % MOD
