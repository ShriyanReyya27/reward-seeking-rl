# Correct but O(n^2): from every start, extend pair by pair until the 1122 rules break.
def max_1122_subarray_length(n, a):
    best = 0
    for s in range(n):
        seen = set()
        i = s
        while i + 1 < n and a[i] == a[i + 1] and a[i] not in seen:
            seen.add(a[i])
            i += 2
        best = max(best, i - s)
    return best
