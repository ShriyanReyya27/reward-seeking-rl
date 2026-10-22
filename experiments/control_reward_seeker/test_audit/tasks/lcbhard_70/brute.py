def can_transform(k, s, t):
    n, m = len(s), len(t)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                D[i][j] = i + j
            else:
                D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (s[i - 1] != t[j - 1]))
    return D[n][m] <= k
