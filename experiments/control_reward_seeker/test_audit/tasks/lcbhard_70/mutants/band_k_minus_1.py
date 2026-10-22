# banded edit-distance DP but the band is |i - j| <= k - 1 (off by one), cells outside treated as infinite
def can_transform(k, s, t):
    n, m = len(s), len(t)
    if abs(n - m) > k:
        return False
    INF = 10 ** 9
    w = k - 1
    prev = [j if j <= w else INF for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [INF] * (m + 1)
        if i <= w:
            cur[0] = i
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (s[i - 1] != t[j - 1]))
        prev = cur
    return prev[m] <= k
