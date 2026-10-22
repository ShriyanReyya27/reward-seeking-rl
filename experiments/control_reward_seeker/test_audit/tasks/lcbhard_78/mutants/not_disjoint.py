# ignores disjointness: counts ordered pairs of nonempty subsequences with equal gcd as cnt[g]^2
from math import gcd
MOD = 10**9 + 7
def count_subsequence_pairs(nums):
    cnt = {0: 1}
    for v in nums:
        nc = dict(cnt)
        for g, w in cnt.items():
            k = gcd(g, v); nc[k] = (nc.get(k, 0) + w) % MOD
        cnt = nc
    return sum(w * w for g, w in cnt.items() if g) % MOD
