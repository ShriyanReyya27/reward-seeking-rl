from itertools import permutations


def count_valid_permutations(n, requirements):
    cnt = 0
    for p in permutations(range(n)):
        ok = True
        for e, c in requirements:
            inv = sum(1 for i in range(e + 1) for j in range(i + 1, e + 1) if p[i] > p[j])
            if inv != c:
                ok = False
                break
        cnt += ok
    return cnt % (10 ** 9 + 7)
