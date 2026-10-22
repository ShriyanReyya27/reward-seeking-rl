# longest run of equal adjacent pairs, ignoring that each value may appear only twice
def max_1122_subarray_length(n, a):
    best = 0
    for s in (0, 1):
        l = s
        i = s
        while i + 1 < n:
            if a[i] != a[i + 1]:
                l = i + 2
            else:
                best = max(best, i + 2 - l)
            i += 2
    return best
