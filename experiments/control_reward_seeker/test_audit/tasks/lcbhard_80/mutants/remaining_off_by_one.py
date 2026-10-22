# digit DP: remaining free bits after a set bit computed as L - i instead of L - i - 1
from math import comb
def count_k_reducible(s, k):
    L = len(s)
    steps = [0] * (L + 3)
    for c in range(2, L + 3):
        steps[c] = 1 + steps[bin(c).count('1')]
    cnt = [0] * (L + 2)
    ones = 0
    for i, ch in enumerate(s):
        if ch == '1':
            r = L - i
            for c in range(ones, min(ones + r, L) + 1):
                cnt[c] += comb(r, c - ones)
            ones += 1
    return sum(cnt[c] for c in range(1, L + 1) if steps[c] <= k - 1) % (10**9 + 7)
