# greedily takes the largest power of two that fits in the remaining length, ignoring alignment
def divide_into_good_sequences(L, R):
    res = []
    l = L
    while l < R:
        size = 1 << ((R - l).bit_length() - 1)
        res.append((l, l + size))
        l += size
    return res
