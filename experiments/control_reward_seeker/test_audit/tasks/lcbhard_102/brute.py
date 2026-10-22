from itertools import permutations


def kth_alternating_permutation(n, k):
    alts = [list(p) for p in permutations(range(1, n + 1))
            if all((p[i] + p[i + 1]) % 2 == 1 for i in range(n - 1))]
    return alts[k - 1] if k <= len(alts) else []
