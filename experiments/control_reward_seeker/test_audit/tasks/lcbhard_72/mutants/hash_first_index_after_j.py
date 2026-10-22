# pairs i<j then looks up x-a_i-a_j in a value->first-index map, requiring that index > j
def find_triple_sum(n, x, a):
    first = {}
    for idx, v in enumerate(a):
        first.setdefault(v, idx)
    for i in range(n):
        for j in range(i + 1, n):
            k = first.get(x - a[i] - a[j])
            if k is not None and k > j:
                return (i + 1, j + 1, k + 1)
    return None
