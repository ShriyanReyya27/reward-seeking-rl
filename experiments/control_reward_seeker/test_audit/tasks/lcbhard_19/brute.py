# enumerate all permutations of 1..AB-1 and check the three conditions literally
from itertools import permutations
from functools import lru_cache


def _lis(seq):
    best = [1] * len(seq)
    for i in range(len(seq)):
        for j in range(i):
            if seq[j] < seq[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best) if seq else 0


@lru_cache(None)
def _count(A, B):
    N = A * B - 1
    c = 0
    for p in permutations(range(1, N + 1)):
        if _lis(p) != A or _lis([-x for x in p]) != B:
            continue
        for n in range(0, N + 1):
            q = p + (n + 0.5,)
            if _lis(q) == A and _lis([-x for x in q]) == B:
                c += 1
                break
    return c


def count_special_permutations(A, B, M):
    return _count(A, B) % M
