# moves the left pointer to last[v]+2 without checking last[v] is inside the window (pointer can jump backwards)
def max_1122_subarray_length(n, a):
    best = 0
    for s in (0, 1):
        l = s
        last = {}
        i = s
        while i + 1 < n:
            if a[i] != a[i + 1]:
                l = i + 2
            else:
                v = a[i]
                if v in last:
                    l = last[v] + 2
                last[v] = i
                best = max(best, i + 2 - l)
            i += 2
    return best
