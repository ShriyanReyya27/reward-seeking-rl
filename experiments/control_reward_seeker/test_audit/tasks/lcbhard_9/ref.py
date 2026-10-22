from math import comb
def count_ways_to_stair(k: int) -> int:
    # after j up-jumps Alice is at 2^j minus the number d of downs; downs are non-consecutive
    # and interleave the j jumps in C(j+1, d) ways.
    total = 0
    for j in range(0, 32):
        d = (1 << j) - k
        if 0 <= d <= j + 1:
            total += comb(j + 1, d)
    return total
