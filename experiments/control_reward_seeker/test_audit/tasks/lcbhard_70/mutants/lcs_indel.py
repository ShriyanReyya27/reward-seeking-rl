# distance = len(s) + len(t) - 2*LCS (insert/delete only; a replacement counted as two operations)
def can_transform(k, s, t):
    n, m = len(s), len(t)
    if abs(n - m) > k:
        return False
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(max(1, i - k), min(m, i + k) + 1):
            cur[j] = prev[j - 1] + 1 if s[i - 1] == t[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return n + m - 2 * prev[m] <= k
