# two-pointer over pairs starting at even indices only (forgets subarrays starting at odd index)
def max_1122_subarray_length(n, a):
    best = 0
    l = 0
    last = {}
    i = 0
    while i + 1 < n:
        if a[i] != a[i + 1]:
            l = i + 2
        else:
            v = a[i]
            if last.get(v, -1) >= l:
                l = last[v] + 2
            last[v] = i
            best = max(best, i + 2 - l)
        i += 2
    return best
