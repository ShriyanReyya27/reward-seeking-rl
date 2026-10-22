# treats x=1 as needing an operation / excludes it (only counts x >= 2)
from math import comb
def count_k_reducible(s, k):
    L = len(s)
    steps = [0] * (L + 2)
    for c in range(2, L + 2):
        steps[c] = 1 + steps[bin(c).count('1')]
    cnt = [0] * (L + 1)
    ones = 0
    for i, ch in enumerate(s):
        if ch == '1':
            r = L - i - 1
            for c in range(ones, ones + r + 1):
                cnt[c] += comb(r, c - ones)
            ones += 1
    tot = sum(cnt[c] for c in range(1, L + 1) if steps[c] <= k - 1)
    if int(s, 2) > 1:
        tot -= 1
    return tot % (10**9 + 7)
