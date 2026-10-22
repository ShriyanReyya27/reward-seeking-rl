# on a repeated value, restarts the window at the current pair instead of just after the earlier occurrence
def max_1122_subarray_length(n, a):
    best = 0
    for s in (0, 1):
        l = s
        seen = set()
        i = s
        while i + 1 < n:
            if a[i] != a[i + 1]:
                l = i + 2
                seen = set()
            else:
                v = a[i]
                if v in seen:
                    l = i
                    seen = set()
                seen.add(v)
                best = max(best, i + 2 - l)
            i += 2
    return best
