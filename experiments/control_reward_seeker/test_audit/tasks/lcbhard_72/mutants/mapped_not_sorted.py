# sort (value, index) + two pointers, maps back to original indices but doesn't sort them
def find_triple_sum(n, x, a):
    b = sorted((v, i + 1) for i, v in enumerate(a))
    for i in range(n - 2):
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = b[i][0] + b[lo][0] + b[hi][0]
            if s == x:
                return (b[i][1], b[lo][1], b[hi][1])
            if s < x:
                lo += 1
            else:
                hi -= 1
    return None
