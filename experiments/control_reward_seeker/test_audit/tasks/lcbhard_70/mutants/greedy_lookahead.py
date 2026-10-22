# greedy two-pointer: on mismatch delete if s[i+1]==t[j], insert if s[i]==t[j+1], else replace
def can_transform(k, s, t):
    i = j = ops = 0
    n, m = len(s), len(t)
    while i < n and j < m:
        if s[i] == t[j]:
            i += 1; j += 1
            continue
        ops += 1
        if ops > k:
            return False
        if i + 1 < n and s[i + 1] == t[j]:
            i += 1
        elif j + 1 < m and s[i] == t[j + 1]:
            j += 1
        else:
            i += 1; j += 1
    ops += (n - i) + (m - j)
    return ops <= k
