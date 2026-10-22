from itertools import product


def count_good_arrays(n, m, k):
    c = 0
    for a in product(range(m), repeat=n):
        if sum(a[i] == a[i - 1] for i in range(1, n)) == k:
            c += 1
    return c % (10 ** 9 + 7)
