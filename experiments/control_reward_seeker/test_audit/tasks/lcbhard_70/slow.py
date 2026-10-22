# Correct full O(|s||t|) edit-distance DP with rolling rows, then compare with k.
def can_transform(k, s, t):
    n, m = len(s), len(t)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        a = s[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != t[j - 1]))
        prev = cur
    return prev[m] <= k
