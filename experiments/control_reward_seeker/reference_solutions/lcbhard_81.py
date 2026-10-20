# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
def min_longest_substring(s: str, numOps: int) -> int:
    n = len(s)
    # length 1: alternating pattern
    mism = sum(1 for i, c in enumerate(s) if (ord(c) - 48) != (i & 1))
    if min(mism, n - mism) <= numOps:
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
    return lo
