# block size = lowbit(l) shrunk to fit, but l == 0 is treated as lowbit 1
def divide_into_good_sequences(L, R):
    res = []
    l = L
    while l < R:
        size = (l & -l) if l else 1
        while l + size > R:
            size >>= 1
        res.append((l, l + size))
        l += size
    return res
