from itertools import combinations
from functools import reduce


def max_subsequence_value(nums, k):
    best = 0
    for idx in combinations(range(len(nums)), 2 * k):
        a = reduce(lambda x, y: x | y, (nums[i] for i in idx[:k]))
        b = reduce(lambda x, y: x | y, (nums[i] for i in idx[k:]))
        best = max(best, a ^ b)
    return best
