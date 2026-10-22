from itertools import product
MOD = 10 ** 9 + 7


def sum_of_subsequence_powers(nums, k):
    # each element: 0 = not in T, 1 = in T but not in S, 2 = in S (S subset of T, sum(S) == k)
    tot = 0
    for st in product(range(3), repeat=len(nums)):
        if sum(x for x, s in zip(nums, st) if s == 2) == k:
            tot += 1
    return tot % MOD
