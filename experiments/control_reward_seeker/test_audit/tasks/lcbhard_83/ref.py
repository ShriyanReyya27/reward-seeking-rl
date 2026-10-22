from operator import le


def max_kagamimochi(n, sizes):
    # binary search on K: K pairs exist iff the K smallest fit on the K largest in order
    a = sizes
    dbl = [2 * x for x in a]
    lo, hi = 0, n // 2
    while lo < hi:
        m = (lo + hi + 1) // 2
        if all(map(le, dbl[:m], a[n - m:])):
            lo = m
        else:
            hi = m - 1
    return lo
