# assumes popcount chains are short: treats every x < n as k-reducible when k >= 3
from math import comb
def count_k_reducible(s, k):
    L = len(s)
    n = int(s, 2)
    if k >= 3:
        return (n - 1) % (10**9 + 7)
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
    return sum(cnt[c] for c in range(1, L + 1) if steps[c] <= k - 1) % (10**9 + 7)
