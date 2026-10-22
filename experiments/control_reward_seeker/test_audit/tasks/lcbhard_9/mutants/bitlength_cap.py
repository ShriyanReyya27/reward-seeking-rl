# loops jump count only up to k.bit_length() (misses larger j for small k, e.g. k=0,1)
from math import comb
def count_ways_to_stair(k):
    total = 0
    for j in range(k.bit_length() + 1):
        d = (1 << j) - k
        if 0 <= d <= j + 1:
            total += comb(j + 1, d)
    return total
