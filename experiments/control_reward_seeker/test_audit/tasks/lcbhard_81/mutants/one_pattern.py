# length-1 special case only checks the alternating pattern starting with '0'
def min_longest_substring(s, numOps):
    n = len(s)
    mism = sum(1 for i, c in enumerate(s) if int(c) != i % 2)
    if mism <= numOps:
        return 1
    runs = []
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        runs.append(j - i)
        i = j
    lo, hi = 2, max(runs)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(r // (mid + 1) for r in runs) <= numOps:
            hi = mid
        else:
            lo = mid + 1
    return max(lo, 1) if max(runs) > 1 else 1
