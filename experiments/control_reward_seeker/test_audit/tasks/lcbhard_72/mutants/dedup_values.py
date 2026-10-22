# two pointers over distinct values only (loses triples that reuse a value)
def find_triple_sum(n, x, a):
    first = {}
    for i, v in enumerate(a):
        first.setdefault(v, i + 1)
    vals = sorted(first)
    m = len(vals)
    for i in range(m):
        lo, hi = i + 1, m - 1
        while lo < hi:
            s = vals[i] + vals[lo] + vals[hi]
            if s == x:
                return tuple(sorted((first[vals[i]], first[vals[lo]], first[vals[hi]])))
            if s < x:
                lo += 1
            else:
                hi -= 1
    return None
