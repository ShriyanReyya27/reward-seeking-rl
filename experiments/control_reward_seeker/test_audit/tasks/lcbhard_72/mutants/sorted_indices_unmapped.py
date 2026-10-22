# sort + two pointers, but returns positions in the sorted array instead of original indices
def find_triple_sum(n, x, a):
    b = sorted(a)
    for i in range(n - 2):
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = b[i] + b[lo] + b[hi]
            if s == x:
                return (i + 1, lo + 1, hi + 1)
            if s < x:
                lo += 1
            else:
                hi -= 1
    return None
