from math import gcd
from itertools import product
def count_subsequence_pairs(nums):
    cnt = 0
    for assign in product(range(3), repeat=len(nums)):
        g1 = g2 = 0
        for v, a in zip(nums, assign):
            if a == 1: g1 = gcd(g1, v)
            elif a == 2: g2 = gcd(g2, v)
        if g1 and g2 and g1 == g2:
            cnt += 1
    return cnt % (10**9 + 7)
