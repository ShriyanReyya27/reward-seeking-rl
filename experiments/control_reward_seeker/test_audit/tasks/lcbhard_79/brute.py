from itertools import permutations


def count_balanced_permutations(num):
    c = 0
    for p in set(permutations(num)):
        if sum(int(p[i]) for i in range(0, len(p), 2)) == sum(int(p[i]) for i in range(1, len(p), 2)):
            c += 1
    return c % (10**9 + 7)
