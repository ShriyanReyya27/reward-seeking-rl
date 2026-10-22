# multiplies C(E, k) * C(O, c-k) per digit without shrinking the free even/odd slots
MOD = 10**9 + 7
from math import comb


def count_balanced_permutations(num: str) -> int:
    n = len(num)
    cnt = [num.count(str(d)) for d in range(10)]
    tot = sum(int(c) for c in num)
    if tot % 2:
        return 0
    half = tot // 2
    E = (n + 1) // 2
    O = n - E
    dp = {(0, 0): 1}
    for d in range(10):
        c = cnt[d]
        nd = {}
        for (e, s), v in dp.items():
            for k in range(c + 1):
                if e + k <= E and s + d * k <= half and c - k <= O:
                    key = (e + k, s + d * k)
                    nd[key] = (nd.get(key, 0) + v * comb(E, k) * comb(O, c - k)) % MOD
        dp = nd
    return dp.get((E, half), 0)
