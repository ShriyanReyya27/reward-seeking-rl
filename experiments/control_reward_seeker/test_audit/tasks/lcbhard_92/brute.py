def sum_of_subarray_powers(n, k, a):
    tot = 0
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            tot += s ** k
    return tot % 998244353
