# correct search but returns 0-based indices
def find_triple_sum(n, x, a):
    seen = {}
    for j in range(n):
        for k in range(j + 1, n):
            i = seen.get(x - a[j] - a[k])
            if i is not None:
                return (i, j, k)
        seen.setdefault(a[j], j)
    return None
