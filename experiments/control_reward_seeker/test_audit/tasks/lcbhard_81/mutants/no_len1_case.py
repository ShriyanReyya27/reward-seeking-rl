# binary search with r // (L+1) flips per run for every L, including L=1 (no alternating-pattern special case)
def min_longest_substring(s, numOps):
    runs = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        runs.append(j - i)
        i = j
    lo, hi = 1, max(runs)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(r // (mid + 1) for r in runs) <= numOps:
            hi = mid
        else:
            lo = mid + 1
    return lo
