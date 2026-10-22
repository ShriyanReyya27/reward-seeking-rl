def find_triple_sum(n, x, a):
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if a[i] + a[j] + a[k] == x:
                    return (i + 1, j + 1, k + 1)
    return None
