from itertools import product


def sum_of_scores(n, m):
    tot = 0
    for L in range(1, n + 1):
        for seq in product(range(1, m + 1), repeat=L):
            X = 1
            for x in seq:
                X *= x
            d = 0
            for q in range(1, X + 1):
                if X % q == 0:
                    d += 1
            tot += d
    return tot % 998244353
