# exhaustive over all 2^(n*n) binary matrices, grouped by (row sums, col sums) -> fixed-cell counts.
# n <= 4 computed here in Python; n = 5 (2^25 matrices) precomputed by brute5.c (same algorithm, in C).
_N5 = {0, 5, 9, 10, 12, 13, 15, 16, 17, 19, 21, 25}  # output of: gcc -O2 brute5.c && ./a.out 5
_cache = {}
def _ach(n):
    if n == 5:
        return _N5
    if n in _cache:
        return _cache[n]
    groups = {}
    for m in range(1 << (n * n)):
        rows = tuple(bin((m >> (i * n)) & ((1 << n) - 1)).count('1') for i in range(n))
        cols = tuple(sum((m >> (i * n + j)) & 1 for i in range(n)) for j in range(n))
        key = (rows, cols)
        if key in groups:
            a, o = groups[key]
            groups[key] = (a & m, o | m)
        else:
            groups[key] = (m, m)
    res = {n * n - bin(a ^ o).count('1') for a, o in groups.values()}
    _cache[n] = res
    return res
def matrix_with_k_fixed_elements_exists(n, k):
    return k in _ach(n)
