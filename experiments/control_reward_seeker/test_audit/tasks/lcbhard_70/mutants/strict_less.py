# correct banded DP but returns distance < k instead of <= k
def can_transform(k, s, t):
    n, m = len(s), len(t)
    if abs(n - m) > k:
        return False
    INF = 10 ** 9
    prev = [j if j <= k else INF for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [INF] * (m + 1)
        if i <= k:
            cur[0] = i
        for j in range(max(1, i - k), min(m, i + k) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (s[i - 1] != t[j - 1]))
        prev = cur
    return prev[m] < k
