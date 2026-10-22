# uses C(j, d) (slots between jumps) instead of C(j+1, d)
from math import comb
def count_ways_to_stair(k):
    total = 0
    for j in range(32):
        d = (1 << j) - k
        if 0 <= d <= j + 1:
            total += comb(j, d)
    return total
