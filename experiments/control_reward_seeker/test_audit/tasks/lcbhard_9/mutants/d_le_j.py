# off-by-one in the number of downs allowed: d <= j instead of d <= j+1
from math import comb
def count_ways_to_stair(k):
    total = 0
    for j in range(32):
        d = (1 << j) - k
        if 0 <= d <= j:
            total += comb(j + 1, d)
    return total
