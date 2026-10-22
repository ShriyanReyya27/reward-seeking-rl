# returns -1 (AtCoder style) instead of None when no triple exists
def find_triple_sum(n, x, a):
    seen = {}
    for j in range(n):
        for k in range(j + 1, n):
            i = seen.get(x - a[j] - a[k])
            if i is not None:
                return (i + 1, j + 1, k + 1)
        seen.setdefault(a[j], j)
    return -1
