from itertools import product
def count_valid_colorings(n, m):
    c = 0
    for t in product(range(m), repeat=n):
        if all(t[i] != t[(i + 1) % n] for i in range(n)):
            c += 1
    return c % 998244353
