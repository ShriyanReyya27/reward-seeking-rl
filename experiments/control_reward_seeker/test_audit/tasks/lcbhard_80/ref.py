from math import comb
MOD = 10**9 + 7
def count_k_reducible(s: str, k: int) -> int:
    L = len(s)
    steps = [0] * (L + 2)
    for c in range(2, L + 2):
        steps[c] = 1 + steps[bin(c).count('1')]
    # cnt[c] = #x in [0, n) with popcount c
    cnt = [0] * (L + 1)
    ones = 0
    for i, ch in enumerate(s):
        if ch == '1':
            r = L - i - 1
            for c in range(ones, ones + r + 1):
                cnt[c] += comb(r, c - ones)
            ones += 1
    total = 0
    for c in range(1, L + 1):
        if steps[c] <= k - 1:
            total += cnt[c]
    return total % MOD
