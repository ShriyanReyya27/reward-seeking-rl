def _ok(t):
    m = len(t)
    if m % 2 == 0:
        return False
    h = (m + 1) // 2
    return t[:h - 1] == "1" * (h - 1) and t[h - 1] == "/" and t[h:] == "2" * (m - h)


def max_eleven_twentytwo_subsequence(s, l, r):
    sub = s[l - 1:r]
    n = len(sub)
    best = 0
    for mask in range(1, 1 << n):
        t = "".join(sub[i] for i in range(n) if mask >> i & 1)
        if _ok(t):
            best = max(best, len(t))
    return best
