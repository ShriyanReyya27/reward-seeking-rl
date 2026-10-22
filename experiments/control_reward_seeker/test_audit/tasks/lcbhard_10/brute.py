from itertools import product


def count_valid_strings(K, C):
    letters = [i for i in range(26) if C[i] > 0]
    tot = 0
    for L in range(1, K + 1):
        for t in product(letters, repeat=L):
            if all(t.count(i) <= C[i] for i in letters):
                tot += 1
    return tot % 998244353
