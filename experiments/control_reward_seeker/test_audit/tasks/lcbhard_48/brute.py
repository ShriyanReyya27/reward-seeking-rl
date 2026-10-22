# all k-subsets by index; power = min |a-b| over all pairs
from itertools import combinations


def sum_of_powers(nums, k):
    t = 0
    for c in combinations(nums, k):
        t += min(abs(a - b) for a, b in combinations(c, 2))
    return t % (10 ** 9 + 7)
