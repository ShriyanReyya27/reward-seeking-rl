# greedy from the left pairing a[i],a[i+1]; on a mismatch restart the window at i+1 (known real model mistake)
def max_1122_subarray_length(n, a):
    best = 0
    i = 0
    start = 0
    last = {}
    while i + 1 < n:
        if a[i] == a[i + 1]:
            v = a[i]
            if v in last and last[v] >= start:
                start = last[v] + 2
            last[v] = i
            best = max(best, i + 2 - start)
            i += 2
        else:
            i += 1
            start = i
    return best
