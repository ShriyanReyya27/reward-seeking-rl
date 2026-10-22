# uses strict inequality (top < half of bottom)
def max_kagamimochi(n, sizes):
    a = sizes
    lo, hi = 0, n // 2
    while lo < hi:
        m = (lo + hi + 1) // 2
        if all(2 * a[i] < a[n - m + i] for i in range(m)):
            lo = m
        else:
            hi = m - 1
    return lo
