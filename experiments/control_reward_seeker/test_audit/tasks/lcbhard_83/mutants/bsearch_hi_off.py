# binary search upper bound (n-1)//2 instead of n//2
def max_kagamimochi(n, sizes):
    a = sizes
    lo, hi = 0, (n - 1) // 2
    while lo < hi:
        m = (lo + hi + 1) // 2
        if all(2 * a[i] <= a[n - m + i] for i in range(m)):
            lo = m
        else:
            hi = m - 1
    return lo
