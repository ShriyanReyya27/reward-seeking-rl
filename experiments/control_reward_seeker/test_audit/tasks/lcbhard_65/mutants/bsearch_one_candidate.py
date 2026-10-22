# binary search for the first '/' where ones-before >= twos-after, without checking the previous '/'
def max_eleven_twentytwo_subsequence(s: str, l: int, r: int) -> int:
    sub = s[l - 1:r]
    n = len(sub)
    p1 = [0] * (n + 1)
    p2 = [0] * (n + 1)
    for i, c in enumerate(sub):
        p1[i + 1] = p1[i] + (c == "1")
        p2[i + 1] = p2[i] + (c == "2")
    ps = [i for i, c in enumerate(sub) if c == "/"]
    if not ps:
        return 0
    ones = [p1[p] for p in ps]
    twos = [p2[n] - p2[p + 1] for p in ps]
    lo, hi = 0, len(ps) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ones[mid] >= twos[mid]:
            hi = mid
        else:
            lo = mid + 1
    return 2 * min(ones[lo], twos[lo]) + 1
